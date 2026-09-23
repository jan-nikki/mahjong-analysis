# 立直宣言牌別の待ち強度分析

## 目的と対象

`data/processed/riichi-waits-v1`のうち2020〜2025年、鳳凰卓東南戦・東場・
赤あり四人麻雀の成立リーチを対象に、立直宣言牌と宣言直後の待ち強度の関連を
調べる。観測単位は成立リーチ1 recordとし、年別でも同じ比較を行う。

## 宣言牌分類

分類は排他的とし、ドラ切りを最優先する。

1. `dora`: 赤5、または`start_kyoku.dora_marker`から定まる表ドラ。
2. `19`: ドラではない数牌1・9。
3. `28`: ドラではない数牌2・8。
4. `37`: ドラではない数牌3・7。
5. `46`: ドラではない数牌4・6。
6. `5`: ドラではない数牌5（赤5は必ず`dora`）。
7. `honor`: ドラではない字牌。

赤牌とドラ表示牌は通常牌種へ正規化して照合する。基盤datasetには局途中の
追加ドラ表示牌が保存されていないため、槓ドラ切りはこの版の`dora`には含めない。
裏ドラは宣言時に見えておらず、分類対象外である。

## 待ち強度

`riichi_wait_quality.evaluate_wait_quality()`を正本として、次を個別に保存する。

- `contains_suji`: standard/ryanmen、またはノベタン系の筋待ちを含む。
- `contains_ryanmen`: 両面解釈を1つ以上含む。
- `good_wait`: 自己所有分を除く重複なしの形式待ち枚数が5枚以上。

上記のいずれかを満たすrecordを`strong_wait`とする。現行定義では両面は
`contains_suji`にも含まれるが、ユーザー指定の比較指標として別に保存する。

`weak_wait`は`strong_wait`でなく、形式待ち枚数が4枚以下で、全待ち解釈が
`tanki / kanchan / penchan / shanpon`のいずれかであるrecordとする。
強い待ちの判定を常に先に行うため、辺張解釈を含んでも5枚以上なら強い。
国士単騎等、強い待ちでも上記弱形でもないrecordは`other_wait`へ残す。
3区分は排他的で全recordを分割する。

形式待ち枚数は他家の手牌・河、本人の河、ドラ表示牌、実山残枚数を控除しない。
したがって実際の山残りや和了率を表す指標ではない。

## 集計と検証

全体、年度、exact宣言打牌番号について、全record、非ドラ全体、宣言牌7分類を
保存する。率は整数countから計算し、分母0は`null`とする。各sliceに次を保存する。

- record数、強い・弱い・その他の件数と率。
- `contains_ryanmen / contains_suji / good_wait`の件数と率。
- 自己所有分を除く形式待ち枚数分布。
- 各待ち形を1つ以上含むrecord数。複合形は複数の形へ計上するため非排他的。

## 宣言牌と同じ筋列に待ち牌を含むrecord

宣言牌が数牌のとき、同色で数字差が3または6のformal wait tileを1つ以上含む
recordを`same_suji_line_wait`とする。たとえば5切りに対する2・8待ち、4切りに
対する1・7待ち、6切りに対する3・9待ちである。これは宣言牌と待ち牌の位置関係
だけを示し、意図的な筋ひっかけ、個別牌の安全度、放銃率を意味しない。

各sliceで次を保存する。

- `same_suji_line_wait`: 全recordを分母とする該当件数・率。
- `weak_and_same_suji_line`: 全recordを分母とする弱形かつ該当の件数・率。
- `weak_and_not_same_suji_line`: 全recordを分母とする弱形かつ非該当の件数・率。
- `same_suji_line_share_within_weak_wait`: 弱形を分母とする該当率。
- `weak_wait_rate_within_same_suji_line`: 該当recordを分母とする弱形率。
- `weak_wait_rate_excluding_same_suji_line`: 非該当recordを分母とする弱形率。
- 弱形かつ該当recordの待ち形別件数。1recordに複数解釈があり得るため非排他的。

## 非ドラ数牌の実rank集計

ドラ優先分類後の非ドラ数牌だけを対象に、宣言牌の実rank 1〜9で主表を作る。
赤5と局開始時の表ドラは`dora`へ送られるため、この主表には含めない。rank主表は
排他的で、各recordを1回だけ計上する。各rankで次を保存する。

- `record_count`。
- `same_suji_line_wait`の件数・全rank recordに対する率。
- `weak_and_same_suji_line`の件数。
- `same_suji_line_wait`該当record内の弱形率。
- `weak_and_not_same_suji_line`の件数。
- 非該当record内の弱形率。
- `strong_wait / weak_wait / other_wait`の件数・率。

次のrank集約が宣言牌分類とStrengthAccumulator全体で完全一致しなければ失敗する。

- rank 1 + rank 9 = `19`
- rank 2 + rank 8 = `28`
- rank 3 + rank 7 = `37`
- rank 4 + rank 6 = `46`
- rank 5 = `5`

## 宣言牌rank×待ち牌rank

同色で数字差が3または6の次の18組を保存する。

`1→4, 1→7, 2→5, 2→8, 3→6, 3→9, 4→1, 4→7, 5→2, 5→8,`
`6→3, 6→9, 7→1, 7→4, 8→2, 8→5, 9→3, 9→6`

各組で、該当record数、宣言牌rank全体に占める率、該当record内の強形・弱形・
その他、弱形の嵌張・辺張・双碰・単騎フラグ、形式待ち枚数分布を保存する。
弱形の形フラグは、その組の待ちrankに対応する待ち解釈を対象とする。同じ待ちrankに
複数解釈があれば複数形へ計上するため非排他的である。

1recordが同じ筋列の2種類の対象待ちrankを含む場合、rank主表と
`same_suji_line_wait`には1回だけ計上し、rank×wait rank表では両方へ計上する。
したがって18組の件数は相互排他的ではなく、合計を主表件数と比較してはならない。

rank主表と18組表は、全体・年度別・exact宣言打牌番号別の各sliceへ保存する。

annual datasetのsize/SHA256、DTO、年度行数を検証する。全recordの既存待ち品質は
`riichi-wait-quality-v1.json`の同じ年度・宣言打牌番号と完全一致させる。
分類と強弱3区分がそれぞれ全recordを分割することを検証する。
強い待ち件数は既存quality baselineの`contains_suji OR good_wait`から独立に導出し、
辺張追加の前後で不変であることを検証する。年度・exact宣言打牌番号・宣言牌分類の
mergeがそれぞれ上位sliceに一致することも検証する。
非ドラ数牌rankから5つの数牌分類を再構成できることも各sliceで検証する。

各宣言牌分類からstable key
`(relative_source_path, start_kyoku_line, reach_line)`の先頭3件を保存する。
元MJAIの物理行を直接参照し、局、成立リーチ、宣言牌、ドラ表示牌を目視できる
ようにする。
