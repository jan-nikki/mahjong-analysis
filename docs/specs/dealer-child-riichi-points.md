# 親子別・成立リーチ平均打点分析仕様（第1回）

## 目的

天鳳鳳凰卓の四人打ち東南戦について、東場で成立したリーチを親と子に分け、
和了時の平均打点、和了率、ツモ／ロン割合をMJAI原本から集計する。

第1回の中心的な問いは次の2点とする。

1. 子のリーチ和了時平均打点は、一般に言われる6,500点程度なのか。
2. 親の平均打点は子の約1.5倍という制度上の差から、どの程度ずれるのか。

6,500点の出典と定義は記事作成時に別途確認する。本分析では6,500点を
外部benchmarkとして数値比較するが、出典確認前に既存研究の確定値とは扱わない。

## 今回扱わないもの

次は後続記事の対象とし、第1回の主要結論へ混ぜない。

- 平和、ドラ、赤ドラ、裏ドラ、一発などの役・翻構成
- 宣言時の待ち形、待ち枚数、愚形率
- 親リーチ後の他家の押し引き、安全牌選択、オリ打ち
- 親子差の因果推論

MJAIの`hora`には得点差分と裏ドラ表示牌はあるが、符・翻・役名はない。
役構成を扱うには和了時までの手牌再生と独立した点数計算が必要なため、
第1回の実装には含めない。

## データと対象範囲

- 元データ: `NikkeTryHard/tenhou-to-mjai` release `v2.0.0`
- 入力: `data/raw/YYYY/*.mjson`
- ルール: filenameのrule codeが`00a9`
- 赤牌: 最初の`start_game.aka_flag is True`
- 対象局: `start_kyoku.bakaze == "E"`
- 主期間: 2020〜2025年
- 長期確認: 2009〜2025年
- 観測単位: 成立リーチ1件
- 親: `reach.actor == start_kyoku.oya`
- 子: `reach.actor != start_kyoku.oya`

`.db`は入力にも中間保存にも使用しない。集計コードは年度別のMJAIを
直接ストリーミング走査し、集計JSON、Markdown、および成立リーチ1件1行の
監査用gzip JSON Linesを出力する。監査用データは集計の再検算とMJAI原本への
追跡に用い、大量生成物としてGitHubにはcommitしない。

通常リーチ、ダブルリーチ、追っかけリーチをすべて含める。同じ局で複数人が
成立リーチした場合はactorごとに1件とする。

主集計は全成立リーチを対象とする。感度分析として、actorの最初の打牌で、かつ
それ以前にチー・ポン・大明槓・暗槓・加槓がないダブルリーチを除いた
「通常リーチのみ」の同一指標も併記する。

## 成立リーチ

既存の`match_reach_sequence()`と同じ定義を用いる。

```text
reach -> 同一actorのdahai -> 同一actorのreach_accepted
```

3イベントは隣接していなければならない。宣言牌に`hora`または`ryukyoku`が
発生して`reach_accepted`まで到達しなかった宣言は成立リーチに含めない。

軽量集計では手牌や待ちを再計算しない。その代わり、production runでは
`data/processed/riichi-waits-v1/manifest.json`の年度別次項目と完全一致させる。

- `scanned_files`
- `target_games`
- `east_kyokus`
- `established_riichis`

このmanifestは母集団件数の独立検算だけに用い、親子分類、局結果、打点の
分析入力には用いない。不一致はskipせずrun全体を失敗させる。

## 局結果

局末の`hora`または`ryukyoku`イベントを結果とする。結果イベントは
`end_kyoku`直前に連続していなければならない。

各成立リーチを次の排他的3区分へ分類する。

- `win`: 同じactorの`hora`が存在する
- `other_win`: `hora`はあるが、そのactorは和了していない
- `draw`: `ryukyoku`で終了した

ダブロン・トリプルロンは各`hora`を独立した和了として扱う。成立リーチ者が
複数和了者の1人なら、そのリーチは`win`である。

和了方法は次のとおり判定する。

- `tsumo`: `hora.actor == hora.target`
- `ron`: `hora.actor != hora.target`

## 打点の定義

主指標の`hand_points`は、本場と供託を除いた和了手本体について、和了者以外が
支払った点数の合計とする。子のツモは親1人と子2人の支払い合計、親のツモは
子3人の支払い合計である。

各`hora.deltas`から次のように求める。単独和了ではその和了者、複数ロンでは
MJAI上で最初の`hora`だけが本場を受け取るものとする。

```text
payer_loss = -sum(和了者以外の負のdelta)
hand_points = payer_loss - (本場受取者なら 300 * honba、そうでなければ 0)
```

ロンでも通常は放銃者1人だけが負になるが、責任払い等に備えて全支払者の
負のdeltaを合計する。本場はロンなら放銃者が1本300点、ツモなら3人が
各100点を支払うため、本場を受け取る和了者について合計`300 * honba`を除く。
天鳳のダブロンでは積み棒・供託は上家取りであるため、2人目以降の`hora`から
本場を再度引かない。

供託は支払者の負のdeltaへ含まれず、和了者の正のdelta側だけへ加算されるため、
上式には入らない。

補助値`settlement_gain`として`hora.deltas[actor]`も保存する。これは本場・供託を
含む和了イベント上の増加点であり、成立時に既に支払った本人のリーチ棒は
差し引かない。主たる「平均打点」には使用しない。

## 集計指標

親・子それぞれについて次を保存する。

- 成立リーチ数
- 和了、他家和了、流局の件数
- 和了率と95% Wilson信頼区間
- ツモ和了数、ロン和了数
- 和了内ツモ割合と95% Wilson信頼区間
- 成立リーチ当たりツモ和了率、ロン和了率
- ツモ和了・ロン和了それぞれの平均打点と打点分布
- `hand_points`合計、平均、中央値、最小、最大、分布
- 平均`hand_points`の95% cluster-robust信頼区間
- 平均`settlement_gain`
- 成立リーチ1件当たり和了点
- 満貫以上率、跳満以上率

満貫以上・跳満以上の基準は実際の親子別和了点を使う。

| 区分 | 満貫以上 | 跳満以上 |
|---|---:|---:|
| 親 | 12,000 | 18,000 |
| 子 | 8,000 | 12,000 |

平均打点の信頼区間は1牌譜をclusterとする。各牌譜内の和了数`n_g`と
打点合計`s_g`について、全体平均を`m`として次のsandwich分散を用いる。

```text
G / (G - 1) * sum((s_g - m * n_g) ** 2) / total_wins ** 2
```

実装では1-pass集計のため、`sum(s_g^2)`、`sum(n_g*s_g)`、`sum(n_g^2)`を
年度結果へ保持する。clusterが2未満、または和了が0件なら信頼区間は`null`とする。

## 比較

期間ごとに最低限次を出力する。

- 親平均打点と子平均打点
- `親平均 / 子平均`
- 子平均と6,500点の差
- 子平均の95%信頼区間が6,500点を含むか

単純な1.5倍との比較は記述的なものとする。低打点の100点単位切り上げや
和了手構成があるため、`親平均 / 子平均 - 1.5`だけで原因を断定しない。
同一和了手を親・子として再計算する反実仮想比較は第2回で行う。

## 出力

production CLIの既定出力は次とする。

- `outputs/dealer-child-riichi-points/summary-v1.json`
- `outputs/dealer-child-riichi-points/summary-v1.md`
- `outputs/dealer-child-riichi-points/records-v1.jsonl.gz`

JSONは次を含む。

- dataset identityとscope
- 打点定義
- 選択期間の合計
- 2020〜2025年小計（全年度が選択されている場合）
- 2009〜2025年小計（全年度が選択されている場合）
- 年度別結果
- 6,500点benchmarkとの差
- 親子別打点分布
- 通常リーチのみの感度分析

監査用JSON Linesには、年度、`YYYY/ファイル名`、牌譜内の局index、局番号、本場、
親・actor、通常／ダブルリーチ、start_kyoku・reach・reach_acceptedの原本行番号、
各eventの局内index、局結果、和了時はhoraの原本行番号と局内index・ツモ／ロン・
本場受取有無・打点を保存する。

### 年度並列実行と原子的公開

production CLIは`--workers N`を受け取り、既定値は1とする。`N`は正の整数で
なければならない。複数年度を選んだ場合は最大`min(N, 対象年度数)` processで、
1年度を1 taskとしてMJAI原本から独立に走査する。Windowsのspawn方式で実行可能な
top-level workerを使い、得点判定と年度集計はsingle-process時と同じ関数を用いる。
共有cancellation eventはspawn時のworker initializerで渡す。いずれかの年度が失敗した
場合は親processがeventをsetし、実行中workerは次のsource fileを開く前に中断する。
処理中の1 source fileは完了させるが、残りの年度全走査を待たずに終了する。

各workerはファイル名順に年度内を処理し、次を完了してから年度結果を親processへ返す。

- raw file数とcanonical cohortの年度別4件数を検証する
- 成立リーチ1件につき1行を、その年度専用の一時gzip JSON Linesへ書く
- 書いた行数が年度集計の`established_riichis`と一致することを検証する
- gzip headerのfilenameを空、`mtime=0`、compression levelを9としてbytesを決定的にする

親processはtaskの完了順にかかわらず`YearlyRiichiPointResult`を年度昇順へ並べ、同じ順で
年度gzip memberをbyte連結する。連結gzipはRFC 1952の複数member streamであり、Pythonの
`gzip.open()`で全年度を連続して読めるものとする。公開前に連結streamを末尾まで読み、
総行数が全年次の`established_riichis`合計と一致することを検証する。このため、
`workers=1`と`workers>1`で集計内容、レコード順、gzip bytesは同一である。

JSON、Markdown、連結gzipはすべて各targetと同じfilesystem上の一時fileへ完成させる。
公開時は既存の3 targetを同じdirectory内のbackupへ移してから`os.replace()`し、3件の
いずれかの置換が失敗した場合は、それまでに置換済みのtargetも含めて3件すべてをrun前の
状態へ戻す。run用の年度member、未公開の一時file、成功後のbackupは削除する。CLIは
使用するworker数、年度内の処理file数、および各年度の完了file数・record数を標準出力へ
表示する。

本番結果を検証後、確定した小さなJSONを`research/results/`へ保存する。

## 検証

人工MJAIで最低限次をテストする。

- 親ロン、子ロン、親ツモ、子ツモ
- 本場と供託を除いた`hand_points`
- 非和了リーチ、流局
- ダブロンで成立リーチ者も和了する場合
- ダブロンの2人目から本場を重複控除しないこと
- ダブルリーチ分類と通常リーチのみ感度分析
- 未成立リーチを除外すること
- 親子分類
- 東場以外と非対象対局の除外
- 不正な`deltas`、結果イベント列、孤立`reach_accepted`の拒否
- 複数年度の結合後も件数・分布・cluster momentsが一致すること
- 同一構成のclusterで分散が厳密に0になる場合の数値安定性
- JSONとMarkdownの整合性
- 複数期間を出すMarkdown tableが途中で分断されないこと
- 監査用gzip JSON Linesから元ファイル・局・event位置を復元できること
- 2 workerの複数年度runで年度昇順のレコード内容、連結gzipの可読性、再実行時の
  gzip bytes一致を確認すること
- 年度検証が失敗した場合に既存の最終成果物を変更せず、一時fileを残さないこと
- 2 workerの一方が失敗した場合、共有eventによって他方がsource file境界で中断し、
  元の年度失敗を原因として報告すること
- 3成果物の途中置換を故意に失敗させ、すべての既存targetがrun前のbytesへ戻ること

本番run後は、親子×ツモ／ロン×本場有無から実牌譜を抽出し、元MJAIの
`start_kyoku`、`reach`、`reach_accepted`、`hora.deltas`を目視確認する。
