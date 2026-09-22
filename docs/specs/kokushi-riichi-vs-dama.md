# 国士無双テンパイ：即リーチとダマの比較仕様

## 目的

2009〜2025年の天鳳鳳凰卓・四麻半荘DBから、国士無双テンパイになった局を
和了・非和了を問わず抽出し、最初の判断を次の3群に分類する。

- `immediate_riichi`: 最初の国士テンパイを作った打牌でリーチを宣言
- `delayed_riichi`: 最初はダマにし、後の国士テンパイ打牌でリーチを宣言
- `dama`: 国士テンパイ後に国士の形からリーチを宣言せず局終了

## 観測単位

1行は「牌譜×局×プレイヤー」の最初の**打牌後13枚の国士テンパイ**とする。
同じプレイヤーがテンパイを維持・解消・再テンパイしても重複行を作らない。

配牌時点ですでにテンパイしていても、自分の最初の打牌前に和了した場合は、
リーチかダマかを選べる判断点がないため対象にしない。

## リーチ選択の扱い

選択を調べるため、`REACH step=2`まで進んだ成立リーチだけに限定しない。
`REACH step=1`と宣言牌を確認できれば、宣言牌で放銃した場合もリーチ選択に含める。

成立の有無は別フィールドに保存する。

- `riichi_declared`: 国士テンパイからリーチを宣言したか
- `riichi_established`: 同じ宣言が`REACH step=2`まで進んだか

このため、従来の成立国士リーチ604件にあった生存者選択を避けられる。

## 保存項目

識別情報、判断時点の状態、後の方針、局結果を同じレコードに保存する。

- `year`, `date`, `log_id`, `kyoku_index`, `who`, `decision_tj`, `url`
- `turn`, `bakaze`, `kyoku_number`, `honba`, `dealer`
- `scores`, `actor_score`, `actor_rank`
- `wait_type`, `wait_group`, `waits`
- `visible_wait_counts`, `unseen_wait_counts`, `furiten`
- `prior_riichis`, `any_call`, `opponent_open_hands`, `opponent_called_hands`
- `remaining_draws_estimate`, `riichi_eligible`
- `strategy`, `riichi_turn`, `riichi_delay_turns`, `riichi_reach_type`
- `riichi_wait_type`, `riichi_waits`, `riichi_declared`, `riichi_established`
- `outcome`, `won`, `win_method`, `winners`, `from_who`, `draw_type`
- `point_delta`

`unseen_wait_counts`は山残りではない。自分の手牌にも公開情報にも存在しない枚数で、
他家手牌と王牌を含む。

`remaining_draws_estimate`は70からその時点までのツモイベント数を引いた値である。

## 主比較

主比較は、最初の判断時点における次の2群とする。

- 即リーチ
- 最初はダマ（後リーチを含む）

これは「最初の国士テンパイで即リーチを選ぶか」を問う比較である。
後リーチは治療方針の交差に相当するため、即リーチ対最後までダマの比較も感度分析として出す。

リーチ不可能な判断点（1000点未満、推定残りツモ4枚未満）は主比較から除く。

## 集計

和了率だけでなく、次を群別に集計する。

- ロン率、ツモ率
- 本人放銃率
- 他家ツモ・他家間ロン
- 流局率
- 局収支（利用可能な`sc`がある場合）

生の比較に加え、共通支持のある層内で構成比をそろえる。

1. 待ち種別×巡目帯×未見枚数
2. 上記＋親子×先行リーチ有無×フリテン×他家副露有無

標準化後の差も観察研究であり、記録されていない手牌価値・対局者判断などは調整できない。
因果的に「リーチが有利」と断定しない。

## 実行

```powershell
.venv\Scripts\python.exe analysis/find_kokushi_tenpai_decisions.py `
  --workers 12 --progress `
  --output outputs/kokushi-tenpai-decisions-2009-2025.json

.venv\Scripts\python.exe analysis/analyze_kokushi_riichi_vs_dama.py
```
