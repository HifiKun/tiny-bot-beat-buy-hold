// Tiny trend bot — EMA crossover on hourly bars (agent-fun/bars-replay/v1).
//
// Strategy: per symbol, hold when the trend is confirmed — the fast EMA (12h)
// sits above the slow EMA (48h), the latest close sits above the fast EMA,
// and the 24h return is positive — and sit in cash otherwise. The extra
// filters keep the bot out of choppy, directionless markets where a bare
// crossover whipsaws (the dev window is a falling market: buy-and-hold loses
// 8.7% there, and a bare 8/24 crossover lost 6.5% on 51 churned trades).
// Capital is split equally across the symbols currently trending, each capped
// at 25% of equity, and entries scale in: half the target once the entry
// conditions have held 2 bars, the full target once they have held 6, so a
// breakout that fades fast only ever costs half a position. Positions are
// rebalanced only when they drift more than 1.5% of equity from target, to
// keep fees low. A drawdown guard exits everything if equity falls 12% below
// its peak and pauses new entries until the drawdown recovers under 6%.
//
// The bot is fully deterministic: every decision comes from the bars it has
// seen, never from the clock, the environment or randomness. It speaks the
// line protocol itself: one JSON line in per batch (hello, then batches, then
// done), exactly one reply line out per batch echoing its seq with the orders
// decided on that bar. Orders decided on bar k fill at bar k + 1's open.

import { createInterface } from 'node:readline';

type U64 = string;

interface HelloSymbol { symbol: string; mint: string; price_scale: number; qty_scale: number }
interface Hello {
  type: 'hello'; protocol: string; seed: U64; symbols: HelloSymbol[];
  capital: { quote: string; quote_scale: number; quote_units: U64 };
  bar_interval: string; first_close_ms: number;
  caps: { max_open_positions: number; max_orders_per_bar: number; max_weight_bps: number; shorts: boolean; leverage: boolean };
  fees: { fee_bps: number; slippage_bps: number; min_notional_quote_units: U64 };
  latency_bars: number; warmup_bars?: number;
}
interface Bar { symbol: string; t: number; o: U64; h: U64; l: U64; c: U64; v: U64 | null; o_depth: U64[]; c_depth: U64[] }
interface Position { symbol: string; qty: U64; cost_quote_units: U64 }
interface Account { cash_quote_units: U64; positions: Position[]; equity_quote_units: U64 }
interface Batch { type: 'batch'; seq: number; t: number; bars: Bar[]; fills: unknown[]; account: Account }
interface Done { type: 'done'; seq: number; reason: string; account: Account }
interface BuyOrder { side: 'buy'; client_id: string; symbol: string; notional_quote_units: U64; max_slippage_bps: number }
interface SellOrder { side: 'sell'; client_id: string; symbol: string; qty: U64; max_slippage_bps: number }
type Order = BuyOrder | SellOrder;

const FAST_SPAN = 12;
const SLOW_SPAN = 48;
const MOMENTUM_BARS = 24; // the 24h return must be positive to enter
const CONFIRM_BARS = 2; // the entry conditions must hold 2 bars in a row
const FULL_BARS = 6; // ... and 6 bars in a row for the full-size entry
const MAX_WEIGHT = 0.25; // 25% of equity per asset (caps.max_weight_bps is 2500)
const REBALANCE_BAND = 0.015; // only trade when off target by 1.5% of equity
const MIN_TRADE_USDC = 2; // skip dust trades (the market's min notional is 1 USDC)
const EXIT_DD = 0.12; // exit everything past a 12% drawdown from peak equity
const RESUME_DD = 0.06; // resume entries once the drawdown is under 6%
const MAX_SLIPPAGE_BPS = 300;
const USDC_RAW = 1_000_000n; // quote units per USDC

interface SymState {
  priceScale: number;
  qtyScale: number;
  emaFast: number | null;
  emaSlow: number | null;
  history: number[]; // recent closes, oldest first, capped at MOMENTUM_BARS + 1
  confirmBars: number; // consecutive bars the entry conditions have held
}

const syms = new Map<string, SymState>();
let cash = 0n;
let equity = 0n;
let peak = 0n;
let paused = false;
let warmupBars = 0;
let orderNo = 1;
const heldQty = new Map<string, bigint>();

function ema(prev: number | null, price: number, span: number): number {
  if (prev === null) return price;
  const k = 2 / (span + 1);
  return price * k + prev * (1 - k);
}

function priceUsdc(sym: SymState, closeRaw: string): number {
  return Number(closeRaw) / 10 ** sym.priceScale;
}

// What `qty` of the symbol is worth at `closeUsdc`, in quote (USDC raw) units.
function valueRaw(sym: string, qty: bigint, closeUsdc: number): bigint {
  const s = syms.get(sym);
  if (s === undefined) return 0n;
  const units = Number(qty) / 10 ** s.qtyScale;
  return BigInt(Math.floor(units * closeUsdc * 1_000_000));
}

// Raw asset quantity worth `valueUsdc` at `closeUsdc`, truncated.
function qtyForValue(sym: string, valueUsdc: number, closeUsdc: number): bigint {
  const s = syms.get(sym);
  if (s === undefined || closeUsdc <= 0) return 0n;
  return BigInt(Math.max(0, Math.floor((valueUsdc / closeUsdc) * 10 ** s.qtyScale)));
}

// Decide the orders for one batch: the hello line already set up the symbols,
// and this batch's bars are folded into the EMAs first.
function decide(batch: Batch): Order[] {
  const orders: Order[] = [];
  if (batch.seq < warmupBars) return orders; // warm-up bars: trading is closed

  // Fold the new bars into the EMAs and remember this bar's closes.
  const closes = new Map<string, number>();
  for (const b of batch.bars) {
    const s = syms.get(b.symbol);
    if (s === undefined) continue;
    const p = priceUsdc(s, b.c);
    closes.set(b.symbol, p);
    s.emaFast = ema(s.emaFast, p, FAST_SPAN);
    s.emaSlow = ema(s.emaSlow, p, SLOW_SPAN);
    s.history.push(p);
    if (s.history.length > MOMENTUM_BARS + 1) s.history.shift();
  }

  // Account state from this batch.
  cash = BigInt(batch.account.cash_quote_units);
  equity = BigInt(batch.account.equity_quote_units);
  if (equity > peak) peak = equity;
  heldQty.clear();
  for (const p of batch.account.positions) heldQty.set(p.symbol, BigInt(p.qty));

  // Drawdown guard: pause entries past a 12% drawdown, resume under 6%.
  const drawdown = peak > 0n ? Number(peak - equity) / Number(peak) : 0;
  if (drawdown > EXIT_DD) paused = true;
  else if (paused && drawdown < RESUME_DD) paused = false;

  // Entry needs the trend confirmed three ways: the fast EMA sits above the
  // slow EMA, the latest close sits above the fast EMA, and the 24h return
  // is positive. Any one alone whipsaws in chop. But an exit needs only the
  // EMA cross to flip: a position is kept while fast stays above slow even
  // if the momentum filters lapse, so a brief flat day does not churn it.
  // (A quicker exit — dropping a position as soon as its close falls below
  // the fast EMA — was tried and lost more: -409 on 37 churned trades.)
  const trending: string[] = [];
  const holdable: string[] = [];
  for (const [sym, s] of syms) {
    const close = closes.get(sym);
    if (s.emaFast === null || s.emaSlow === null || close === undefined) continue;
    if (s.emaFast <= s.emaSlow) continue;
    holdable.push(sym);
    // Entry needs the momentum conditions to hold CONFIRM_BARS bars in a row,
    // so a one-bar spike does not trigger a buy.
    let cond = false;
    if (s.history.length >= MOMENTUM_BARS + 1) {
      const dayAgo = s.history[0];
      cond = close > s.emaFast && close > dayAgo;
    }
    s.confirmBars = cond ? s.confirmBars + 1 : 0;
    if (cond && s.confirmBars >= CONFIRM_BARS) trending.push(sym);
  }

  const bandRaw = BigInt(Math.floor(Number(equity) * REBALANCE_BAND));
  const minRaw = BigInt(MIN_TRADE_USDC) * USDC_RAW;

  // Sells first: when flat (guard paused or nothing holdable), sell everything.
  if (paused || holdable.length === 0) {
    for (const [sym, qty] of heldQty) {
      if (qty <= 0n) continue;
      const close = closes.get(sym);
      if (close === undefined) continue;
      if (valueRaw(sym, qty, close) < minRaw) continue;
      orders.push({ side: 'sell', client_id: `o${orderNo++}`, symbol: sym, qty: qty.toString(), max_slippage_bps: MAX_SLIPPAGE_BPS });
    }
    return orders;
  }

  // Target per trending asset: an equal split, capped at 25% of equity.
  const perAssetRaw = BigInt(Math.floor(Number(equity) * Math.min(MAX_WEIGHT, 1 / trending.length)));

  // Exit symbols whose EMA cross flipped, and trim symbols above target.
  for (const [sym, qty] of heldQty) {
    if (qty <= 0n) continue;
    const close = closes.get(sym);
    if (close === undefined) continue;
    const cur = valueRaw(sym, qty, close);
    if (!holdable.includes(sym)) {
      if (cur >= minRaw) {
        orders.push({ side: 'sell', client_id: `o${orderNo++}`, symbol: sym, qty: qty.toString(), max_slippage_bps: MAX_SLIPPAGE_BPS });
      }
    } else if (cur > perAssetRaw + bandRaw) {
      const excessUsdc = Number(cur - perAssetRaw) / 1_000_000;
      const sellQty = qtyForValue(sym, excessUsdc, close);
      if (sellQty > 0n && sellQty <= qty && cur - perAssetRaw >= minRaw) {
        orders.push({ side: 'sell', client_id: `o${orderNo++}`, symbol: sym, qty: sellQty.toString(), max_slippage_bps: MAX_SLIPPAGE_BPS });
      }
    }
  }

  // Buys: top trending symbols up toward their target, from cash on hand only
  // (a sell's proceeds never fund a buy decided in the same reply). The
  // target scales in: half until the entry conditions have held FULL_BARS.
  let cashForBuys = cash;
  for (const sym of trending) {
    const close = closes.get(sym);
    if (close === undefined) continue;
    const qty = heldQty.get(sym) ?? 0n;
    const cur = valueRaw(sym, qty, close);
    const s = syms.get(sym);
    const scale = s !== undefined && s.confirmBars >= FULL_BARS ? 1 : 0.5;
    const targetRaw = BigInt(Math.floor(Number(perAssetRaw) * scale));
    const delta = targetRaw - cur;
    if (delta <= bandRaw) continue;
    if (delta < minRaw) continue;
    const spend = delta < cashForBuys ? delta : cashForBuys;
    if (spend < minRaw) continue;
    orders.push({ side: 'buy', client_id: `o${orderNo++}`, symbol: sym, notional_quote_units: spend.toString(), max_slippage_bps: MAX_SLIPPAGE_BPS });
    cashForBuys -= spend;
  }

  return orders.slice(0, 16);
}

async function main(): Promise<void> {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim() === '') continue;
    const msg = JSON.parse(line) as Hello | Batch | Done;
    if (msg.type === 'hello') {
      const hello = msg as Hello;
      warmupBars = hello.warmup_bars ?? 0;
      cash = BigInt(hello.capital.quote_units);
      equity = cash;
      peak = cash;
      for (const s of hello.symbols) {
        syms.set(s.symbol, { priceScale: s.price_scale, qtyScale: s.qty_scale, emaFast: null, emaSlow: null, history: [], confirmBars: 0 });
      }
    } else if (msg.type === 'batch') {
      const batch = msg as Batch;
      const orders = decide(batch);
      process.stdout.write(JSON.stringify({ type: 'reply', seq: batch.seq, orders }) + '\n');
    } else if (msg.type === 'done') {
      break;
    } else {
      throw new Error(`unknown message type ${(msg as { type: string }).type}`);
    }
  }
  process.stdin.destroy();
}

main().catch((err: unknown) => {
  console.error(err instanceof Error ? err.stack ?? err.message : String(err));
  process.exit(1);
});
