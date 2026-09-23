# 親リーチ後の他家防御行動分析仕様（第3回）

## 目的

第1回で観測した親子別リーチのロン・ツモ構成差について、先制して成立した親リーチと
子リーチの後で、他家の打牌と行動がどのように異なるかを記述する。
記事タイトルは「親リーチはなぜロンされにくい？ ツモ増加と他家の防御行動を牌譜から調べる」
を想定し、最低限次を調べる。

- 他家が現物、筋、無筋、字牌をどの割合で切ったか
- 他家のツモ切り、鳴き、後続リーチがどの程度あったか
- リーチ者に何回の自摸と、子の他家に何回の打牌機会が生じたか
- 非フリテンのリーチに対する子の打牌1回当たりロン発生率
- 上記の親子差が、巡目、実際の待ち、席順、開閉状態、応答順、安全牌数の共通支持内でも
  残るか

本分析は観察研究である。「親だから他家を降ろした」「防御行動がロン率を下げた」という
因果効果は推定しない。特に実際の待ち形・待ち枚数は他家から見えないため、待ちによる層別は
牌譜上の親子構成差とロン機会をそろえるための調整であり、他家が待ちを認識して打牌したという
解釈には用いない。記事中の「なぜ」は、観測された構成差を記述的に分解する問いとして扱う。

## 入力と対象範囲

入力は次の4種類とする。

```text
data/processed/riichi-waits-v1/manifest.json
data/processed/riichi-waits-v1/YYYY.jsonl.gz
data/raw/YYYY/*.mjson
outputs/dealer-child-riichi-points/summary-v1.json
```

- 元データ: `NikkeTryHard/tenhou-to-mjai` release `v2.0.0`
- ルール: 四人麻雀の鳳凰卓東南戦、filename rule code `00a9`、赤牌あり
- 対象局: `start_kyoku.bakaze == "E"`
- 対象年: 2009〜2025年
- 主期間: 2020〜2025年
- 長期確認: 2009〜2025年
- focal観測単位: 1局で最初に成立したリーチ1件
- 打牌観測単位: focal成立後のprimary responderによる`dahai`1件
- responder観測単位: focalリーチとprimary responderの組1件

manifestは`dataset_name == "riichi-waits-v1"`, `schema_version == 1`, full extraction、
2009〜2025年の全17年を要求する。source repository、release、scopeの`rule_code`, `aka_flag`,
`bakaze`、年度、年度別filename・件数・SHA-256も固定する。第1回summaryは
`analysis_name == "dealer-child-riichi-points-v1"`, schema version 2、同じsourceとscope、
同じ17年を要求する。

主分析には通常リーチとダブルリーチを含める。感度分析では、第1回と同じく「actorの最初の
打牌で、それ以前にチー・ポン・大明槓・暗槓・加槓がない」成立リーチをダブルリーチとして
除き、通常リーチだけで全指標を再集計する。通常／ダブルの判定を結果表から差し引いて作らず、
raw eventからfocalごとに判定する。

## focalリーチとresponder

### focalリーチ

各局の成立リーチを`reach_accepted_event_index`順に並べ、最初の1件だけをfocalとする。
成立リーチは既存実装と同じ、隣接した

```text
reach -> 同一actorのdahai -> 同一actorのreach_accepted
```

である。未成立の`reach`をfocalにはしない。同じ局の2件目以降の成立リーチはfocalにせず、
後述の観測終了条件およびresponder行動として扱う。

focal roleは`actor == oya`なら`dealer`、それ以外なら`nondealer`とする。

### primary responder

主要比較では、focal以外の「子」だけをresponderとする。

- focalが親: 他の3人全員
- focalが子: 親を除いた他の子2人

これにより、主要な親子比較のresponder自身は常に子になる。子focalに対する親responderは
primaryから除外するが、打牌、安全牌更新、局進行からは除外しない。親responderの打牌・鳴き・
後続リーチも含め、全3人を`secondary_all_responders`として別の生集計へ出す。子focal時には
親responderもここへ含む。このsecondary集計には共通支持標準化を適用しない。

relative seatは名前へ変換せず、次の整数で保存する。

```text
relative_seat = (responder - focal_actor) % 4
```

値は1、2、3のいずれかである。

## 観測区間とevent境界

観測区間はfocalの`reach_accepted`の**直後**から始まり、次のいずれか早い方で終わる。

1. 次の成立リーチの`reach_accepted`
2. 局末の`hora`または`ryukyoku`

開始event自身と終了eventより後は含めない。次の`reach_accepted`は
`window_end_reason == second_reach_accepted`として記録して終了する。その成立リーチの`reach`と
宣言`dahai`は終了eventより前なので観測に含める。
したがって2軒目のリーチ宣言牌も打牌機会になる。一方、宣言牌でロンされて
`reach_accepted`に到達しなかったリーチはcensorにせず、その宣言`dahai`とロン結果を含める。

`dahai`を処理するときは必ず次の順とする。

1. そのevent直前の安全牌集合、開閉状態、応答ordinalで特徴量と牌分類を確定する。
2. 当該`dahai`に続く局末`hora`列を調べ、focalのロンかを付与する。
3. focalのロンでなければ、その牌を「focalが通した牌」として以後の安全牌集合へ加える。
4. `dahai`後の鳴きや状態変化を、それ以後のdecisionにだけ反映する。

ダブロン・トリプルロンでは局末の連続した全`hora`を調べ、1件でも
`hora.actor == focal_actor`かつ`hora.target == discard_actor`なら当該打牌の`focal_ron`を1とする。
複数`hora`があっても打牌分母・ロン分子は1件である。加槓への槍槓等、直前の子の`dahai`へ
帰属できないfocalロンは局末`terminal_outcome`には保存するが、打牌1回当たりロン率の分子・分母には
入れない。

応答区間の終了理由と実際の局末結果は混ぜない。`window_end_reason`は`hand_end`または
`second_reach_accepted`、`terminal_outcome`は局全体を最後まで読んだ`focal_tsumo`, `focal_ron`,
`other_win`, `draw`のいずれかとする。2軒目成立で応答区間を打ち切った後も局末まで分類し、
局末`hora`件数、multi-ron、成立／未成立2軒目宣言牌上のロン、打牌へ帰属しないfocalロンを
独立した監査整数として保存する。これらの監査値によって応答区間のexposureは延長しない。

区間内のfocal actorの`tsumo` event数を`focal_draw_exposure`とする。区間内のprimary child
responderの`dahai` decision数を`child_discard_exposure`とする。終了eventは数えない。

## 状態再生

### 安全牌集合

牌種比較では赤5を通常5へ正規化する。例えば`5mr`と`5m`は同じ牌種である。

各decision直前の安全牌集合は、次の和集合を重複除去した牌種集合とする。

1. focalの河: focal宣言前から当該decision直前までのfocal自身の全`dahai`
2. focal成立後に他家が切り、当該decisionより前にfocalがロンしなかった牌

現在分類しようとしている牌は、分類が終わるまで集合へ加えない。したがって成立後に初めて
切られた牌を「その打牌自身が通ったから現物」とする先読みは起こらない。チー・ポンされた
打牌も、focalがロンせず通した事実は変わらないため以後の集合へ残す。集合の大きさは
物理枚数でなく、正規化後の異なる牌種数である。

### 安全度分類

各打牌を次の優先順で排他的に1区分へ分類する。

1. `genbutsu`: 正規化牌種がdecision直前の安全牌集合にある
2. `honor_non_genbutsu`: 現物でない字牌
3. `full_suji`: 現物でない数牌で、必要な筋牌がすべて安全牌集合にある
4. `partial_suji`: 現物でない4〜6の数牌で、両側の筋牌の片方だけが安全牌集合にある
5. `unsuji_numbered`: 上記以外の現物でない数牌

同一suitのrankを`r`とすると、必要な筋牌は次とする。

- `r in {1, 2, 3}`: `r + 3`の1種
- `r in {7, 8, 9}`: `r - 3`の1種
- `r in {4, 5, 6}`: `r - 3`と`r + 3`の2種

1〜3および7〜9は必要な1種があれば`full_suji`、なければ`unsuji_numbered`であり、
`partial_suji`にはならない。この分類は待ちへの絶対安全性を意味せず、壁、ワンチャンス、
牌効率、手牌価値等も評価しない。

### フリテン

focalの宣言時フリテンを、正規化したfocal待ち牌集合と、宣言`dahai`までのfocal河の
正規化牌種集合との積が空でないこと、と定義する。

```text
focal_furiten_at_accept = bool(wait_tiles ∩ focal_river_at_accept)
```

ロン率のheadline分母は`focal_furiten_at_accept == false`のfocalに属するprimary child
discardだけとする。これは宣言時フリテンによる構造的なロン不能を除くための固定定義である。
成立後に待ち牌を見逃したことによる動的なリーチ後フリテンを別の選択条件へせず、その経路は
安全牌集合と観測outcomeに反映する。指標名には「宣言時非フリテン」と明記する。

### 開閉状態と行動

`chi`, `pon`, `daiminkan`, `kakan`を経験したactorはopenとする。`ankan`だけではclosedのままと
する。raw eventsを`start_kyoku`から再生し、responderごとに次を保存する。

- `open_at_focal`: focal `reach_accepted`直後の状態
- `current_open`: 各decision直前の状態
- `any_post_riichi_call`: 観測区間内にresponder自身が`chi`, `pon`, `daiminkan`, `ankan`,
  `kakan`のいずれかを1回以上行ったか
- `later_established_riichi`: 観測を終了させた次の`reach_accepted`のactorがそのresponderか

`ankan`は`any_post_riichi_call`には数えるが、open状態にはしない。

primary responderの区間内`dahai`にはresponderごとに1から連番の`response_ordinal`を付ける。
鳴きによる追加打牌も1件として連番を進める。`tsumogiri`はraw `dahai.tsumogiri`のbooleanを
そのまま使い、直前の`tsumo`から推測し直さない。

### focalの待ちclass

wait datasetの導出済み値から、次の優先順で排他的に分類する。

1. `pure_ryanmen`: `is_pure_ryanmen`
2. `contains_ryanmen_other`: pureではなく`contains_ryanmen`
3. `multiwait_other`: ryanmenを含まず`wait_tile_count > 1`
4. `single_other`: 上記以外

実際の待ち牌種数は`wait_tile_count`を使う。待ちclassと待ち枚数は他家の可視情報ではない。

## 集計指標と分母

focal role別に、少なくとも次の生の整数分子・分母と厳密な比率を保存する。比率だけを保存せず、
分母0はJSON `null`、Markdown `N/A`とする。

| 指標 | 分子 | 分母 |
|---|---|---|
| 現物打牌share | `classification == genbutsu`のprimary decision数 | 全primary child discard decisions |
| 無筋数牌share（headline） | `classification == unsuji_numbered`のprimary decision数 | 全primary child discard decisions |
| 無筋条件付きshare（補助） | `classification == unsuji_numbered`のprimary decision数 | 現物でない数牌のprimary decisions |
| ツモ切りshare | `tsumogiri == true`のprimary decision数 | 全primary child discard decisions |
| focalロン率 | 当該打牌からfocalロンになったprimary decision数 | 宣言時非フリテンfocalのprimary child discard decisions |
| 区間内鳴き率 | `any_post_riichi_call == true`のresponder組数 | primary child responder組数 |
| 後続成立リーチ率 | `later_established_riichi == true`のresponder組数 | primary child responder組数 |
| 子打牌exposure | primary child discard decisionsの総数 | focalリーチ数 |
| focal自摸exposure | 区間内focal `tsumo` eventの総数 | focalリーチ数 |

`genbutsu`, `honor_non_genbutsu`, `full_suji`, `partial_suji`, `unsuji_numbered`の5区分すべてについても、全primary
decisionを分母とする件数・shareを出す。合計件数がdecision総数へ一致することを検証する。
all responders secondaryには同じ打牌分類、ツモ切り、宣言時非フリテンfocalロン、鳴き、
後続リーチ、exposureを生集計で出すが、共通支持調整値は作らない。

補助集計として、観測終了理由、focalの局末結果、multi-ron件数、宣言牌ロン件数、
discardへ帰属しないfocalロン件数を保存する。これらをheadlineの分母へ暗黙に混ぜない。

## breakdownと共通支持標準化

### 固定bin

連続・多値変数は実データを見て境界を動かさず、次へ固定する。

- focal turn（`riichi_discard_number`）: `1-5`, `6-8`, `9-11`, `12+`
- wait count: `1`, `2`, `3+`
- response ordinal: `1`, `2`, `3+`
- decision直前safe-kind count: `0-3`, `4-7`, `8-11`, `12+`

relative seatは1、2、3、開閉状態はboolean、wait classは前節の4区分を使う。

生集計は少なくともresponse ordinal、relative seat、responderのopen/closed、focal turn bin、
wait class別に表示する。JSONでは交差表の整数十分統計も保持し、Markdownでは主要な単変量
breakdownを表にする。

### decision指標のstratum

現物、無筋、ツモ切り、宣言時非フリテンfocalロン等、decisionを分母とする指標には次の
完全交差を使う。

```text
(year,
 focal_turn_bin,
 wait_class,
 wait_count_bin,
 responder_relative_seat,
 responder_open_at_focal,
 responder_current_open,
 response_ordinal_bin,
 safe_kind_count_bin)
```

ロン率では宣言時非フリテンのdecisionだけからstratumの分子・分母を作る。他のdecision指標は
全primary decisionを使う。同じstratum keyでも、指標ごとのeligible denominatorから共通支持を
判定する。

### responder指標のstratum

区間内鳴き率と後続成立リーチ率には、1 focal × 1 primary responderを1観測として次を使う。

```text
(year,
 focal_turn_bin,
 wait_class,
 wait_count_bin,
 responder_relative_seat,
 responder_open_at_focal)
```

`current_open`、response ordinal、安全牌数は複数decisionを持つresponderに一意でないため、
responder-level stratumへ入れない。exposure指標はfocal件数を分母とする生の平均だけを示し、
この共通支持標準化の対象外とする。

### 標準化式

focal roleを`r in {D, C}`、stratumを`s`、指標の分子・分母を`y_rs`, `n_rs`とする。期間内で
両roleの`n_rs > 0`であるstratumだけをcommon support `S*`とする。片側だけのstratumを相手側へ
外挿したり、隣接binへまとめたりしない。

```text
w_s = (n_Ds + n_Cs) / sum_{t in S*}(n_Dt + n_Ct)

adjusted_rate_r = sum_{s in S*} w_s * (y_rs / n_rs)
adjusted_difference = adjusted_rate_D - adjusted_rate_C
```

各stratumの整数`y_rs`, `n_rs`を正本とする。生率、coverage、exposure平均等の単純比は整数
`numerator`/`denominator`を保存し、表示用`decimal`は60有効桁、`ROUND_HALF_EVEN`で作った文字列とする。
floatは計算入力に使わない。

調整率と差について、全stratum分母の最小公倍数を持つ巨大なaggregate `Fraction`、aggregate
`numerator`/`denominator`は作らない。各項の整数分子
`(n_Ds + n_Cs) * y_rs`と整数分母`pooled_denominator * n_rs`をPython整数で正確に作ってから、
ambient contextに依存しない`localcontext(Context(prec=60, rounding=ROUND_HALF_EVEN, ...))`内で
Decimal除算・canonical stratum順の加算を行う。JSONの調整値は
`representation == "weighted_sum_of_exact_stratum_counts"`, `decimal`文字列、
`precision_digits == 60`, `rounding == "ROUND_HALF_EVEN"`を持ち、再計算の正本は後述の整数表とする。

各指標について次のcoverageをrole別に出す。

- 全eligible分母`sum_s n_rs`
- common support内分母`sum_{s in S*} n_rs`
- common support外分母とcoverage比
- union stratum数、role別の非空stratum数、common stratum数
- common supportのpooled denominator

`S*`が空、またはpooled denominatorが0なら調整値と差を`null`として`not_estimable`を明示する。
生集計まで失敗させない。片側だけのstratumを落としたことはcoverageで可視化する。

この標準化は共通支持内の記述的比較であり因果推論ではない。focal role以外の交絡を網羅せず、
`current_open`、ordinal、安全牌数にはfocal成立後に生じた状態も含まれる。調整後差を
「親リーチの効果」と呼ばない。

## 期間と感度分析

結果は主分析と通常リーチのみの感度分析について、それぞれ次を作る。

- `selected`: CLIで選択した全年度
- `yearly`: 選択した各年度
- `primary_2020_2025`: 2020〜2025年をすべて選択した場合
- `long_term_2009_2025`: 2009〜2025年をすべて選択した場合

複数年periodの標準化stratumには`year`を残し、年度構成の違いを同じstratumとして潰さない。
通常リーチ感度分析でもfocal選択、decision/responder行、十分統計、common supportを最初から
再構築し、主分析結果からダブルリーチだけを事後的に引かない。

periodごとのmetric objectは生集計、調整表示値、support/coverage要約だけを持ち、stratum rowsを
複製しない。top-level `period_definitions`で各periodの構成年を明示する。

## streaming実装

annual gzipを全件tuple/listへ実体化しない。recordのcanonical順序を検証しながら
`relative_source_path`ごとの連続groupだけを保持し、groupが変わったら対応するraw MJAIを1回だけ
開いて全局を再生する。同一source pathを後で再登場させてはならない。保持メモリは1 source
file内のwait recordsを基本単位とし、並列時も`workers * 2 * batch_size` source以内のtasks/results、
各workerの1 source events、固定個数の集計器、固定上限auditに限定する。選択年の全record数に比例する
identity setや全decision列を持たない。

1 source group内では`start_kyoku_line`, `reach_event_index`, `actor`が厳密昇順で、rawのevent位置と
一致しなければならない。同じ局のrecordsから最小`reach_accepted_event_index`をfocalに選ぶ。
rawの先頭`start_game`、filename rule code `00a9`、`aka_flag == true`を検証する。source内の全東場局で
rawから成立リーチを再抽出し、wait recordsとの欠落・余分がないことを双方向に照合する。
後続recordはcensor identityと全成立リーチの整合性検証に使う。focalについては宣言までの全河を
rawから再構成し、牌、正規化牌、ツモ切り、打牌番号、event index、line、宣言牌、鳴かれた情報を
wait recordと一致させる。focalロン時は放銃牌の正規化牌種がwait tilesに含まれることも要求する。
source pathはPOSIX相対pathとして検証し、解決後pathが`raw_root`外へ出る値を拒否する。

raw source単位の処理は純粋なworker境界へ分離する。workerはraw fileの読込、byte SHA-256、必要時の
gzip展開、UTF-8/JSON decode、rule/赤牌判定、全東場局とwait recordsの双方向照合、focal observation
生成、Article 1 reconciliation差分を完結し、親processのbuilder、global digest、counter、出力へは
触れない。親processだけがcanonicalなyear/source path順にworker結果を適用し、raw digestの
`path + NUL + content digest`順、固定上限auditの先頭順、整数集計順を直列実装と一致させる。

`workers > 1`ではWindows `spawn`互換の`ProcessPoolExecutor`を使う。source tasksは既定32件ずつの
固定上限batchとし、同時投入batch数は`workers * 2`以下に制限する。annual gzip iteratorを全件
future化せず、入力wait recordsと完了済みresultの保持量をsource総数から独立にする。futureの完了順で
適用せず、提出順のdeque先頭からだけ結果を取得・検証・適用する。worker例外はlogical source pathを
含めてfail closedとし、未開始futureをcancelして最終成果物を公開しない。

manifest、Article 1 summary、選択annual gzipはbyte chunkでSHA-256を計算する。full runではannual
gzipを末尾まで読み、複数gzip member、UTF-8、JSON object、schema、record数、byte hashをmanifestと
照合する。raw fileはgroup単位で1回だけ読む。

## 出力

既定出力は次の2成果物とする。

```text
outputs/dealer-riichi-defense/summary-v3.json
outputs/dealer-riichi-defense/summary-v3.md
```

JSONは`analysis_name == "dealer-riichi-defense-v1"`, `schema_version == 3`とし、最低限次を含める。

- analysis name、schema version、run mode、selected years、period定義
- source repository/release/scope、入力の論理pathとSHA-256
- manifestのdataset identity、選択annual entryのfilename・件数・SHA-256
- focal/responder/window/safe-set/suji/furiten/open/denominator/standardizationの機械可読な定義
- 読み込んだwait record、source、局、focal、primary/secondary responder、decision、eventの各件数
- 全分類、`window_end_reason`、`terminal_outcome`、terminal補助監査、除外・非対応eventの件数
- period/year、主分析/通常リーチ、focal roleごとの生指標とbreakdown
- decision/responder指標の共通支持調整値、親子差、coverage、role別eligible/common/outside分母、
  role別非空・common・union stratum数、common pooled分母、推定status
- top-level `standardization_sufficient_statistics`の選択年度×感度別整数表
- focal単位のchild discard/focal draw exposure
- all responders secondaryの生集計
- reconciliationの実施状態と一致した期待値・実測値
- 固定上限の監査例

監査例は全decisionを保存せず、各集計sliceについてcanonical source順のfocalを先頭32件まで、
各focal内のdecisionを先頭8件まで保持する。`year`, `source_path`, `start_kyoku_line`, focal actor,
`reach_accepted_event_index`、responder、decision event index/line、raw/normalized tile、safety class、
decision直前の正規化済みsafe tile kindsとその件数、tsumogiri、open状態、response ordinal、
censor/outcomeを含め、MJAI原本へ戻れるようにする。監査例を得るための乱数・reservoir samplingは
使わない。

Markdownは記事確認用として、selected、利用可能なprimary/long-term、年度別について次を示す。

- focal・responder・decision数と終了理由
- 親子別のheadline生指標、exposure、親子差
- 安全度5分類、response ordinal、relative seat、open/closed、turn、wait class別の主要値
- 共通支持調整値、調整後差、coverage、共通/除外/union stratum数とrole別common/total/outside分母
- 通常リーチ感度分析
- all responders secondary
- 非因果解釈と、actual waitが他家から不可視である注意

`standardization_sufficient_statistics`はdecision/responder unitごとにdimensions、metric columns、
row layoutのheaderを1回だけ持つ。各tableは`year`, `sensitivity`とcanonical順のrow配列からなり、
rowはstratum dimensions、role、全metricの`numerator`, `denominator`を持つ。あるstratumで非eligibleの
metricも`0, 0`として明示する。これらの年次表と`period_definitions`だけから、任意periodの生率、
common support、coverage、pooled weights、調整DecimalをJSON単独で再計算できなければならない。
同じstratum keyをmetricごと、periodごとに繰り返さない。

JSONとMarkdownは同じin-memoryの整数十分統計から作り、`sort_keys=True`, compact separators,
UTF-8, `allow_nan=False`で決定的に出力する。timestamp、処理時間、絶対pathを最終成果物へ入れない。

2成果物はtargetと同じfilesystem上の一時fileへ完成させ、serialize後のJSONを再読込してから
1 bundleとして公開する。既存targetをbackupへ移す前にrecovery pathを状態へ登録し、各
`os.replace()`の直後を含む途中失敗や`KeyboardInterrupt`等の`BaseException`では、target、stage、
backupの存在状態から両方をrun前のbytesへrollbackする。run前に存在しなかったtargetは削除する。
rollback自体が失敗した場合は旧artifactのbackupを絶対に削除せず、そのpathを例外へ含める。
成功後だけ一時fileとbackupを削除する。

## CLIとpartial smokeの安全策

既定CLIは次の形とする。

```text
uv run python analysis/analyze_dealer_riichi_defense.py --all
uv run python analysis/analyze_dealer_riichi_defense.py --years 2025
uv run python analysis/analyze_dealer_riichi_defense.py --all --workers 8
```

`--all`と`--years YEAR [YEAR ...]`の一方を必須とし、重複年と範囲外年を拒否する。manifest、
annual root、raw root、第1回summary、出力directoryはpath optionで上書き可能にする。
論理pathは絶対pathと分離してmetadataへ保存する。既定入力・出力pathはcurrent working directoryで
なく、CLI file位置から解決したproject rootを基準にする。

`--workers N`と`--batch-size N`は正整数だけを受け付け、既定値はそれぞれ1と32とする。
`workers == 1`はprocess poolを作らない直列pathを使う。worker数とbatch sizeは実行戦略であって研究結果の
metadataではないため、`run_mode`や他の最終成果物へ記録しない。同じ入力ならworker数・batch sizeに
かかわらずJSON/Markdown bytesが一致しなければならない。進捗は親processがcanonical順に適用した
source数だけを表示する。

開発時のsmoke用に`--max-files N`を設ける。これはちょうど1年度について、wait datasetに出現する
sourceをraw-root-relative POSIX path順へ並べた先頭N件だけを対象にする。正の整数以外、複数年度との併用、
`--all`との併用を拒否する。annual wait streamとはsource pathの順序を使ってmerge joinし、同じ
raw fileを複数回開かない。

partial smokeでは次を必須とする。

- `run_mode == "partial_smoke"`をJSON/Markdownへ明記する
- manifest/source identityと、実際に読んだ範囲のrecord/raw整合性は検証する
- annual全件数・full annual hash、第1回summaryの年度/period件数とのreconciliationは実施せず、
  各項目を`skipped_partial_run`として列挙する
- canonicalな既定output pathへの書込みを禁止し、project-root基準のJSON/Markdown targetを
  それぞれresolved pathで比較して、両方に明示的な非既定pathを要求する
- partial結果をproduction結果や`research/results/`へ昇格させない

実装後はまず2025年の10,000 raw filesについて`workers=1`と`workers=8`を別directoryへ出力し、
wall time、peak memory、JSON/Markdown hash、source grouping、監査例、rollbackを確認する。改善が乏しい
場合は4または12 workersも比較する。その成功はfull production runの完了とは扱わない。

## 入力検証とreconciliation

full runは少なくとも次をfail closedで検証する。

- manifest、annual file、第1回summaryのidentity、scope、全17年が一致する
- manifest自身のSHA-256と選択annual gzipのstreaming SHA-256をmetadataへ保存する
- annual gzipを末尾まで読み、manifestの年度別`output_records`, `established_riichis`, SHA-256と
  一致する
- wait record schema、bool/int区別、enum、null条件、導出済みwait値を既存loaderで検証する
- annual recordのcanonical order、source pathの年度、source groupの連続性を検証する
- raw pathがroot内にあり、recordのstart/reach/dahai/accepted lineとevent index、actor、局情報、
  河、待ちrecord identityが一致する
- raw gameがrule `00a9`、赤牌ありで、対象recordが東場に属する
- 各局の成立リーチ抽出結果がwait recordsと過不足なく一致する
- 選択年度ごとの全成立リーチ数がmanifestと第1回summaryに一致する
- rawから再計算した親子、win/other-win/draw、ツモ/ロン、および通常リーチsubsetの整数件数が
  第1回summaryの年度結果と一致する
- 第1回summaryの各periodは、その全構成年を処理した場合に年度整数十分統計から再構成した値と
  一致する。summaryの`selected`はCLI選択年がsummary自身のselected yearsと同一の場合だけ比較し、
  それ以外のCLI選択年についてsummaryの別periodだと誤認しない
- focal数が対象局数以下、primary responder数が親focalで3倍・子focalで2倍になる
- safety classの5件数合計、role別decision合計、responder/focal exposure合計が相互に一致する
- focalロンdecisionは待ち牌とnormalized discardが一致し、対応`hora`のactor/targetが一致する
- 次の成立`reach_accepted`より後のeventが混入せず、その宣言`dahai`は混入する
- 通常リーチ感度分析にdouble-riichi focalが0件である
- JSONとMarkdownが同じresult objectから作られ、再実行でbytesが一致する

平均やrateの照合は保存済み小数を計算入力にせず、整数分子・分母から厳密に再計算する。
不一致recordをskipして分析を続けない。エラーにはlogical source path、start line、kyoku、actor、
event indexを可能な限り含める。

## 検証計画

人工MJAIと人工wait recordsで最低限次をテストする。

- 親focalでは子3人、子focalでは子2人がprimaryになり、親responderはsecondaryだけになる
- 2軒目の宣言`dahai`を含め、その`reach_accepted`で厳密にcensorする
- 未成立リーチの宣言牌ロンをdecisionとfocalロンへ含める
- ダブロン・トリプルロンでfocalを見落とさず、1 decisionを重複加算しない
- 赤5と通常5が同じsafe kindおよび待ち牌として扱われる
- 現在の打牌をsafe setへ加える前に`genbutsu`判定する
- focal河と成立後に通った牌からsafe setが更新される
- `full_suji`, `partial_suji`, `unsuji`, `honor`の全境界、とくに1〜3、4〜6、7〜9
- relative seat 1/2/3
- focal/responderの`open_at_focal`, `current_open`を鳴き前後で更新し、`ankan`だけはclosedを保つ
- responderの鳴きと後続成立リーチを正しいresponder組へ付与する
- 宣言時フリテンの定義、非フリテンロン分母、focal draw/child discard exposure
- response ordinalと全固定binの境界
- focal wait classの優先順位
- 観測終了のfocalツモ、focalロン、他家和了、流局、後続リーチ
- decision/responder標準化の手計算、片側stratum除外、pooled weight、coverage
- 5,000桁超の合成分母になる多数stratumでも巨大aggregate Fractionを作らず、serialize/readback後に
  年次整数rowから同じ60桁Decimalを再計算できること
- period metricにstrata表がなく、年次十分統計がunit共通headerと全metric列を持ち、非eligibleが
  `0/0`、複数年periodを全rowから再構築できること
- common supportが0のとき調整値だけ`null`になり、生集計は残る
- 通常リーチ感度分析からdouble riichiを除く
- source groupingでraw fileを1回だけ開き、group再登場、逆順、重複を拒否する
- concatenated gzip、壊れたJSON/schema、manifest hash/count/path不一致を拒否する
- 第1回summaryとの件数・結果・通常リーチ不一致を拒否する
- `--max-files`がreconciliationを無効化し、既定outputを拒否する
- `--workers`, `--batch-size`の正整数検証と、既定直列path
- `workers=1`と`workers=8`でJSON/Markdown bytesが一致する
- future完了順を意図的に逆転してもcanonical順で適用され、in-flight batch数が`workers * 2`以下になる
- worker例外がsource contextを保持し、pending futureをcancelして成果物を公開しない
- 監査例が上限32でcanonical先頭順になり、原本eventへ戻れる
- JSON/Markdownの決定性と、2成果物の途中公開失敗・`KeyboardInterrupt`からのrollback

production run後は、親子focal × response ordinal × safety class × ron有無 × responder開閉から
監査例を抽出し、MJAI原本の`start_kyoku`, focal `reach`/`dahai`/`reach_accepted`, responder
`dahai`, call, censor event, `hora`を目視確認する。特に「現在の牌を先に現物化していないこと」、
2軒目宣言牌、未成立宣言牌ロン、multi-ronを重点確認する。

## 今回扱わないもの

- 他家の手牌価値、向聴数、受入れを使った押し引きの最適性評価
- 同じ局面で親リーチと子リーチを交換する因果的反実仮想
- プレイヤー強度、卓内選択、点棒状況、順位価値の完全な調整
- 壁、ワンチャンス、跨ぎ筋、裏筋、間四軒などを含む高度な安全度モデル
- リーチ者の待ちを他家が知っていたという解釈
- 次の成立リーチ後の多軒リーチ局面
- 非放銃後の将来EV、off-policy評価
- confidence interval、bootstrap、仮説検定

結果は「観測された親／子focal後の他家行動差」「指定strataの共通支持内で標準化した記述差」
と表現し、「親リーチの因果効果」「他家が実際の待ちを読んだ効果」とは表現しない。
