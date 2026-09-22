# 対子落としに見える河の後の成立リーチ分析

## 目的と対象

`data/processed/riichi-waits-v1` の2009〜2025年、鳳凰卓東南戦・東場・
赤あり四人麻雀の成立リーチ10,706,714件を対象に、本人河の
「対子落としに見える連続打牌」と宣言直後の待ちの性質を比較する。
観測単位と正式な分母は成立リーチ1 recordであり、該当ペア数や
`wait_details`数ではない。通常、ダブル、追っかけを含む既存datasetの
母集団を変更しない。

本分析は河の外見上の関連を記述するものであり、実際に手牌内の対子を
意図的に落としたこと、押し引きの優劣、因果効果、放銃率を示さない。

## 主パターン

同じactorの`actor_discards_before_riichi`にある隣接2打について、次を
すべて満たすものを1該当ペアとする。

- `discard_number`が連続する。
- `normalized_tile`が一致する。赤5と通常5は同じ牌種である。
- 2枚目の`tsumogiri`が`false`である。
- 1枚目は手出し、ツモ切りのどちらも許容する。

1枚目が手出しならA、ツモ切りならBとする。2枚目がツモ切りなら非該当。
数牌と字牌の両方を対象とする。宣言打牌が2枚目の場合も含む。
河は本人の打牌だけで隣接を判定し、間に他家のイベントがあるかは問わない。
鳴かれた打牌も除去せず元の番号と順序を維持する。

`was_called`、`call_type`、`called_by_actor`、`call_event_index`、宣言直後の
手牌、待ち情報は主パターンの選択条件に使用しない。局末まで更新された
未来のcall metadataが分類へ逆流してはならない。

## record分類と距離

全該当ペアを見て、recordを次の排他的4区分へ1回だけ計上する。

- `none`
- `a_only`
- `b_only`
- `both`

たとえば「ツモ切り→手出し→手出し」の同一牌3連続はBとAの2ペアを持つが、
recordは`both`へ1件だけ計上する。該当ペア数は別の診断分布へ保存する。

距離用代表は、2枚目の`discard_number`が最大、すなわち最後に完成した
該当ペアとする。

```text
distance = riichi_discard_number - representative second discard_number
```

宣言打牌が代表ペアの2枚目なら0、パターンなしはnullである。代表選択は
待ちラベルに依存しない。A/B区分は代表だけでなく全該当ペアから決める。

## 待ち指標

`src/mahjong_analysis/riichi_wait_quality.py`の`evaluate_wait_quality()`を
唯一の正本として再利用し、定義を複製しない。

主要指標は次のとおり。

- `contains_suji`: standard/ryanmenまたは同色のstandard/tanki解釈2種が
  3または6離れるノベタン系を1つ以上含む。
- `good_wait`: 宣言直後の純手牌とfixed meldを正規化して自己所有分を除き、
  重複しないformal wait tileの`4 - self_owned`合計が5枚以上。
- `contains_ryanmen`: 既存baselineとの連続性を確認する補助指標。
- `self_excluded_wait_copies`: 上記の自己所有分を除いた待ち枚数分布。

formal wait、5枚目待ち、赤5正規化の意味は既存wait-quality仕様を変更しない。

## 比較群、分母、差

全期間、2020〜2025小計、年度、exact宣言打牌番号について、最低限次を保存する。

- 全record
- パターンあり（`a_only + b_only + both`）
- パターンなし（`none`）
- Aのみ
- Bのみ
- AとBの両方

各record-level率の分母はその群のrecord数である。分母0は`null`とする。
主要差はパターンあり率からパターンなし率を引いたpercentage pointsで保存する。
overallは年度率や巡目率の平均ではなく、整数countを合算して再計算する。

遅い宣言ほどパターンの出現機会が増えるため、overallの単純差だけを
パターン固有の効果と解釈しない。exact宣言打牌番号別の両群比較を正式結果へ
含める。「各打牌番号で成立したリーチのうち何割が該当するか」であり、
その打牌番号でリーチが発生する確率ではない。

パターンありrecordについて、代表ペアから宣言までのexact distance別にも
同じ待ち指標を保存する。距離別件数の合計はパターンあり件数に一致する。

## ストリーミングと決定論性

manifest記載の17年度ファイルだけをmanifest順に使用する。年度gzip JSONLを
全件listへ載せず、DTO validationと集計を同じstreaming passで行う。
年度単位でWindows互換のprocess workerを使用できる。年度結果は年度順、
巡目と距離は整数昇順へmergeし、worker完了順に依存させない。

処理前にannual fileのsize/SHA256をmanifestと照合し、年度行数とtotalを
manifest `output_records`へ照合する。全recordを分割前に合算した整数値は、
同じinput manifestを持つ`riichi-wait-quality-v1.json`のoverall、年度、
exact巡目の対応値と完全一致しなければならない。

## 監査サンプル

最大24 unique recordsを、小さいstable key
`(relative_source_path, start_kyoku_line, reach_line)`から決定論的に選ぶ。
監査理由は固定順で統合し、最終samples配列はstable key昇順で保存する。

固定カテゴリは`none`、`a_only`、`b_only`、`both`、宣言打牌で代表ペア完了、
赤5を含む該当ペア、複数該当ペア、観測時点以前に鳴かれた本人打牌を含むrecord
とする。各カテゴリの先頭3件を候補とし、鳴かれた実例があればそのstable key
最小1件を優先確保した後、固定カテゴリ順で各1件目、各2件目、各3件目を
round-robin追加し、stable keyで重複排除して24件で止める。

鳴かれた実例の優先条件は`call_event_index is not null`かつ
`call_event_index <= reach_accepted_event_index`である。局末の
`was_called=true`だけでは選ばない。監査情報は主分類や分母を変更しない。

## 出力

正式な小規模JSONを`research/results/toitsu-drop-analysis-v1.json`へ保存する。
schema version、input manifest SHA256、dataset/analysis generator情報、定義、
検証結果を含める。既存baselineやcanonical datasetを変更しない。

Work向けには次を作る。

```text
outputs/work/toitsu-drop-analysis/
  handoff.md
  article-brief.md
  result-summary.json
  work-prompt.txt
  figures/
```

## 必須不変条件とテスト

- 4分類の合計が全record数。
- `pattern_present + no_pattern == all`。
- ペア数分布の合計が全record数。
- 距離別合計がパターンあり件数。
- 年度合計、巡目合計がoverall。
- formalな待ち指標の各分母が各群record数。
- serial/parallelの内容と順序が一致。
- malformed JSON、年度不一致、行数不一致、worker errorを黙って飛ばさない。
- 手出し→手出し、ツモ切り→手出し、2枚目ツモ切り、非連続、赤5、字牌、
  宣言打牌完了、複数ペア、重複3連続、call metadata非依存を人工期待値で固定。
- 新分析の全record集計が既存wait-quality結果とoverall・年度・巡目で一致。

本番実行後、監査sampleの元MJAIをsource path、局、actor、event位置から
read-onlyで確認する。少数監査は全件の正しさの証明とは扱わない。
