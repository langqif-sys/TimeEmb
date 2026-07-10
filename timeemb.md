# TimeEmb：一种用于时间序列预测的轻量级静态-动态解耦框架

Mingyuan Xia<sup>∗</sup> 吉林大学 xiamy2322@mails.jlu.edu.cn

Zijian Zhang<sup>†</sup> 吉林大学 zhangzijian@jlu.edu.cn

Qidong Liu 西安交通大学 liuqidong@xjtu.edu.cn

Chunxu Zhang 吉林大学 zhangchunxu@jlu.edu.cn

Hao Miao 香港理工大学 hao.miao@polyu.edu.hk

Yuanshao Zhu 香港城市大学 yuanshao@ieee.org

Bo Yang 吉林大学 ybo@jlu.edu.cn

## 摘要

时间非平稳性，即时间序列分布随时间发生变化的现象，对可靠的时间序列预测提出了根本性的挑战。直观上，复杂的时间序列可以分解为两个因素，即时不变（time-invariant）和时变（time-varying）分量，分别表示静态和动态模式。然而，现有方法往往将时变和时不变分量混为一谈，联合学习结合了的长期模式和短期波动，导致在面对分布偏移时表现不佳。为了解决这一问题，我们首次提出了一种用于时间序列预测的轻量级静态-动态分解框架——TimeEmb。TimeEmb 创新性地将时间序列分离为两个互补的分量：（1）时不变分量，通过一个新颖的全局嵌入模块进行捕获，该模块学习跨时间序列的持久表征；（2）时变分量，通过一种高效的频域过滤机制进行处理，该机制受到信号处理中全谱分析的启发。在真实世界数据集上的实验表明，TimeEmb 的性能优于最先进的基线方法，且所需的计算资源更少。我们进行了全面的定量和定性分析，以验证静态-动态解耦的有效性。该轻量级框架还可以通过简单的集成来改进现有的时间序列预测方法。为了便于复现，代码已开源至 https://github.com/showmeon/TimeEmb。

## 1 引言

边缘设备和移动感知的普及产生了海量的时间序列数据，从而赋能了各种实际应用 [58, 51, 15, 26]。在这项研究中，我们专注于时间序列预测，它在能源管理 [11]、交通系统 [57, 7] 和金融市场 [44] 等关键领域的决策制定中发挥着至关重要的作用。

传统的统计方法（例如 ARMA [2]）利用滑动平均技术来建模时间依赖性。随着神经网络的发展，深度学习方法彻底改变了时间模式的提取，并提供了更优异的性能。这些方法包括捕获序列动态的循环神经网络（RNN）模型 [13, 35]、获取层次特征的卷积神经网络（CNN）模型 [8, 18]，以及利用自注意力机制学习长距离依赖关系的基于 Transformer 的模型 [46]。最近，多层感知机（MLP）方法 [56] 展示了其有效性，且与基于 Transformer 的同类方法相比具有更高的效率。

尽管现有方法取得了诸多进展，

![](images/433e40eee762f1997fb671b1dddc829588e82f399a7ab8c61da6ec90aaad113a.jpg)  
图 1：在 Electricity 数据集上的效率和性能对比。

在对复杂的时间依赖性进行建模时，仍然存在一个根本性的挑战，即非平稳性。真实世界的时间序列由于不断演变的趋势和外部干预，往往会表现出动态分布偏移，呈现出高度的非平稳性 [32, 20]。这种动态分布偏移违反了大多数现有预测方法所采用的独立同分布（IID）假设 [9]。这给鲁棒的时间依赖性建模带来了巨大的挑战 [38]，并迫切需要一种能够处理这种非平稳性并学习全面时间依赖关系的新方法。

直观上，时间序列可以看作是两个互补部分的组合：静态时不变分量和动态时变分量 [21, 36, 6]。时不变分量代表时间序列中稳定的长期模式。例如，交通流量通常遵循规律的模式，早晨出现高峰，夜间出现低谷。时变分量则反映了时间序列中的局部波动，例如由于极端天气或交通事故导致的异常交通流量。我们认为，有效解耦这两个分量可以防止模型将短期噪声误认为长期模式。它可以显式地捕获稳定的长期依赖关系和动态的局部依赖关系，从而提高时间序列预测的有效性。

然而，开发此类模型绝非易事。一般来说，时间序列解耦仍有三个主要局限性尚未解决：（1）忽视了长期不变模式的建模。现有的季节性-趋势（seasonal-trend）解耦方法通常通过滑动平均核生成趋势分量，并将剩余部分视为季节性分量 [51, 56, 59, 25]。这主要是通过平滑局部时间序列来实现的 [37]，很难学习到整个时间序列中的全局静态模式。（2）过于严苛的假设。为了追求显式解耦，某些方法依赖于强假设，而这些假设在实际中并不总是成立。例如，CycleNet [14] 假设数据集中存在固定的周期性模式，并使用可学习的循环周期来提取它。然而，这种假设并不总是成立，因为复杂的周期性可能会发生变化或具有不同的长度。此外，依赖预先定义的周期长度会导致灵活性受限且效果不稳定。如果不提供准确的周期长度，它就无法学习周期性。（3）高模型复杂度。自注意力机制的平方复杂度阻碍了实际应用 [51, 59]。如图 1 所示，基于 Transformer 的方法表现出相对较大的模型大小和高昂的训练成本。最近基于频率分析和 MLP 的方法通过更高效的架构在一定程度上缓解了这一问题。然而，在性能和效率之间取得令人满意的平衡仍然非常困难。

为了解决这些问题，我们提出了 TimeEmb，一个轻量级的静态-动态解耦框架。TimeEmb 将原始时间序列分解为时不变分量和时变分量，并进行相应的处理。具体来说，我们引入了一个可学习的时不变嵌入库（embedding bank）来提取静态时不变模式。这些嵌入在整个数据集的所有时间序列片段中是一致的，旨在捕获长期且稳定的时间模式。此外，嵌入库为各个时间步提供特定的嵌入。这使模型能够适应局部数据分布偏移，因为时不变模式在不同的时间步上可能会有所不同。通过从时间序列中分离出时不变分量，我们得到了剩余的时变分量，它展示了动态扰动。频率分析利用复杂信号在频谱中的强度来对其进行描述 [1]，从而呈现出清晰的内在周期性特征。受此启发，我们设计了一个高效的频率滤波器，通过稠密权重（dense weighting）来处理时变分量。基于静态和动态分量的显式分解和并行处理，TimeEmb 取得了最先进的性能。同时，由于其轻量级的架构，它所需的计算资源更少。正如其在图 1 中的最佳位置所示，所提出的 TimeEmb 在性能和效率之间取得了极佳的平衡。

我们的主要贡献总结如下：

• 我们首次提出利用可学习的嵌入库来捕获全局循环特征，同时适应局部分布偏移。

• 我们提出了 TimeEmb，它显式地对时间序列进行解耦，系统地使用可学习的嵌入库处理时不变分量，并通过频率过滤处理时变分量。

• 所提出的 TimeEmb 可以轻松且无缝地作为插件使用，以最小的额外计算成本增强现有方法。

• 在来自不同场景的七个基准数据集上的实验证明了所提出 TimeEmb 的优异性能。与现有最先进的基线方法相比，TimeEmb 在计算和存储方面均非常高效。

## 2 相关工作

基于 Transformer 的时间序列预测。Transformer 已在时间序列预测中展示了强大的序列建模能力 [51, 58, 22]。PatchTST [28] 将序列分割为固定长度的 patch 以进行局部-全局建模，而 iTransformer [22] 和 Informer [58] 降低了注意力复杂度以提高可扩展性。然而，基于注意力的模型仍面临相当大的计算和内存开销 [12, 43]，限制了其在资源受限设置中的部署。相比之下，TimeEmb 利用轻量级的谱模块（即包括一个嵌入库和一个频率滤波器）来实现强大的性能，同时降低了开销。

基于 MLP 的时间序列预测。最近，MLP 方法（例如 TSMixer [4] 和 TimeMixer [49]）在降低复杂度的同时，展示了具有竞争力的预测性能。DLinear [56] 通过分离趋势和残差分量进一步提高了效率。相比之下，TimeEmb 在频域中提供了一个显式的解耦框架，能够同时对时不变和时变模式进行建模，超越了时域 MLP 的表达能力。

基于频率的时间序列预测。最近的研究探索了基于傅里叶的表征，以对周期性进行建模并降低噪声敏感性 [52, 59, 29]。虽然大多数方法采用全局谱分析，但 TimeEmb 引入了一种细粒度的解耦策略：通过嵌入在整个频谱上学习时不变分量，而动态部分则通过可学习的频率调制进行自适应过滤。这种结构化的谱设计扩展了频域建模对复杂时间序列的实用性。

嵌入增强的预测。嵌入策略已被用于编码位置、空间或时间上下文 [28, 39, 10]。例如，STID [39] 和 D2STGNN [41] 使用时空嵌入，而 SOFTS [10] 在通道间共享嵌入。与这些方法不同，TimeEmb 建立了一个可学习的时间嵌入库，用以捕获整个数据集中的全局时不变模式，其中每个嵌入专用于特定时间段，以数据驱动和频率感知的方式建模静态结构。

基于 LLM 的时间序列预测。随着大语言模型的飞速发展，最近的研究通过将时间信号视为序列 token，探索了其在时间序列预测中的潜力 [19, 16]。基于 LLM 的方法受益于强大的泛化和迁移能力，从而能够在不同领域进行零样本（zero-shot）或少样本（few-shot）预测 [17, 24, 48]。然而，这些模型通常是资源密集型的，需要海量的预训练语料库，这使得它们在轻量级或特定领域的应用中并不实用。与这些范式相比，TimeEmb 专注于一种紧凑且有效的解耦机制，在显著降低计算成本 and 训练复杂度的同时，实现了相当的预测精度。

## 3 方法

## 3.1 框架概述

给定具有 $L$ 个时间步和 $D$ 个通道的历史时间序列 $\pmb { X } \in \mathbb { R } ^ { L \times D }$，时间序列预测旨在推断未来 $H$ 个时间步的状态，即 $\widehat { \pmb { X } } \in \mathbb { R } ^ { H \times D }$。

TimeEmb 通过频域中的解耦表征学习来解决时间序列预测问题。其核心思想是将输入序列分解为时不变分量 $X _ { s }$ 和时变分量 $X _ { d }$。

具体来说，我们首先将输入序列 $X$ 转换为其频域表示。

![](images/9fd257de20157c533ad2eff4c2c34363cc0ab5d51c5de7c3ef7d205e6e371824.jpg)  
图 2：TimeEmb 框架概述。

通过傅里叶变换得到 $\overline { { \boldsymbol { X } } }$。然后，我们根据输入时间戳从可学习的嵌入库 $\pmb { { \cal E } }$ 中检索出 $X _ { s }$，以捕获长期稳定的模式。动态部分 $X _ { d }$ 是通过从 $\overline { { \boldsymbol { X } } }$ 中减去 $X _ { s }$ 获得的。为了对复杂的动态进行建模，我们对 $X _ { d } ,$ 应用了一个可学习的频率滤波器 $\mathcal { H } _ { \omega }$，以突出信息丰富的频率并抑制噪声。然后将过滤后的动态和静态分量融合并转换回时域以进行最终预测。这种基于频率的分解使得 TimeEmb 能够以轻量级且可解释的方式高效地捕获周期性结构和瞬态变化。

## 3.2 域变换

从频域的角度审视时间序列数据，可以为其潜在结构提供独特的见解。与时间域（在时域中，模式可能会被噪声或非线性所掩盖）不同，频谱揭示了不同周期分量的分布及其相对能量贡献。将时间序列转换到频域会将其分解为不同的频率分量，从而将复杂的信号描述为具有不同频率和振幅的正弦波和余弦波的线性组合。这一过程有助于揭示在时域中原本模糊不清的潜在周期性和隐藏特征 [42]。

给定离散时间序列 $\pmb { X } \in \mathbb { R } ^ { L \times D }$（为了清晰起见，我们考虑 $D = 1$ 的单变量情况），我们首先进行实例归一化 InstNorm()，以标准化每个实例在每个时间步的分布。然后，可以使用实值快速傅里叶变换（rFFT）[27] 获取其频域表示 $\overline { { \boldsymbol { X } } } \in \mathbb { C } ^ { \boldsymbol { F } \times \boldsymbol { D } }$：

$$
{ \overline { { \pmb { X } } } } [ k ] = \sum _ { n = 0 } ^ { L - 1 } { \pmb { X } } [ n ] e ^ { - j 2 \pi k n / L } , \quad k = 0 , 1 , . . . , F - 1 ,\tag{1}
$$

其中 $j = \sqrt { - 1 }$ 是虚数单位。由于实信号在傅里叶域中具有共轭对称性，因此唯一频率分量的数量为 $F = \lfloor L / 2 \rfloor + 1$，从而实现了无冗余的紧凑表示。

## 3.3 通过嵌入库获取静态分量

现有的对时不变模式进行建模的方法（例如季节性-趋势分解 [51, 56]）通常使用局部平滑方法将时间序列分为趋势分量和残差分量。然而，这种方法仅考虑了输入时间序列中局部稳定和动态的部分，未能发现数据集中长期存在的不变特征。最近，CycleNet [14] 试图通过学习周期性嵌入来解决这一问题，但它依赖于来自专家知识的预定义周期长度，微小的变化就可能导致性能严重下降。

为了解决这些局限性，TimeEmb 提出了一种灵活且可学习的机制，通过时间嵌入库（temporal embedding bank）来捕获跨时间序列共享的长期、循环模式。例如，在交通预测中，我们的目标是捕获循环的每日结构，如典型的上下班高峰期模式。由于日内（intra-day）模式也会随时间而变化（例如，每小时的交通流量波动），我们为每个时间步构建嵌入。

具体来说，我们定义了一个可学习的嵌入库 $\pmb { { \cal E } } \in \mathbb { R } ^ { M \times F \times D }$，它由 $M$ 个嵌入组成，用以保留一天的时不变模式。$M$ 控制着日内特定模式的粒度。例如，当 $M = 2 4$ 时，E 为每小时分配一个嵌入；当 $M = 9 6$ 时，它捕获每 15 分钟的共同模式。为了确保嵌入能够学习到跨时间序列的通用模式，我们利用输入 $X$ 的最后一个时间步作为 $t _ { l a s t }$。该索引使我们能够从 $E$ 中检索嵌入，即 $X _ { s } = E [ t _ { l a s t }$ mod $M ]$。然后，我们从时间序列 $\overline { { \boldsymbol { X } } }$ 中分离出嵌入分量 $X _ { s }$，并得到时变分量 $X _ { d }$，公式如下：

$$
X _ { d } = \overline { { \pmb { X } } } - \pmb { X } _ { s } .\tag{2}
$$

在此操作中，我们从复数 $\overline { { \boldsymbol { X } } }$ 的实部中减去实数 $X _ { s }$，这可以降低嵌入库的计算和存储成本。嵌入库 E 在整个数据集上进行优化，并学习编码在不同日期的同一时间出现的致模式。例如，当 $M = 2 4$ 时，每个嵌入都经过调整，以捕获一天中特定小时 the 平均行为（例如，在 8:00 左右达到峰值，在 23:00 左右达到低谷），从而使模型能够同时表示全局时间结构和局部变化。重要的是，这种嵌入结构是非常灵活的：虽然我们专注于日级周期性，但通过调整 $M$，它可以自然地扩展到对周级或其他常识性周期进行建模，而无需依赖领域知识。这种设计使 TimeEmb 能够学习时不变分量的共享表达性表征，这对于解耦建模和鲁棒的泛化至关重要。

## 3.4 通过频率过滤获取动态分量

为了有效地对动态分量 $X _ { d } ,$ 进行建模，我们在频域中应用了一个可学习的谱滤波器。这一设计的灵感来源于卷积定理 [23]，即时域中的循环卷积等价于频域中的逐元素相乘。因此，频域过滤为在时间信号上实现时不变线性操作提供了一种高效且富有表现力的方法。

我们引入了一个跨通道共享的复值谱调制向量 $\boldsymbol { \omega } \in \mathbb { C } ^ { F \times 1 }$，以选择性地重新加权不同的频段。过滤操作定义为：

$$
\begin{array} { r } { \mathcal { H } _ { \omega } ( X _ { d } ) [ k ] = X _ { d } [ k ] \odot \omega [ k ] , } \end{array}\tag{3}
$$

其中 ⊙ 表示点积。

这一操作可以解释为学习线性时不变（LTI）系统的频率响应函数 [50]。通过端到端地优化 ω，该模型可以灵活地逼近时间信号的线性时不变变换。这为建模各种时间动态提供了理论上的通用性和实践中的灵活性。理论分析可参见附录 A。调制后，过滤后的动态分量与时不变部分 $X _ { s }$ 融合，以恢复完整的频率表示：

$$
\dot { \boldsymbol X } = \mathcal { H } _ { \omega } ( \boldsymbol X _ { d } ) + \boldsymbol X _ { s } .\tag{4}
$$

## 3.5 预测层

我们利用预测层 $f _ { \theta }$，在给定表征 $\dot { X }$ 的情况下生成最终的预测。它可以根据特定要求进行定制。我们在 TimeEmb 中采用了双层 MLP 架构：

$$
\begin{array} { r } { f _ { \theta } ( X ) = W _ { \mathrm { 2 } } ( \mathrm { R e L U } ( W _ { 1 } X + b _ { 1 } ) ) + b _ { 2 } , } \end{array}\tag{5}
$$

其中 $W _ { 1 } \in \mathbb { R } ^ { d \times L }$ 和 $W _ { 2 } \in \mathbb { R } ^ { H \times d }$ 是投影矩阵，H 表示预测步长，而 $b _ { 1 } \in \mathbb { R } ^ { d } , b _ { 2 } \in \mathbb { R } ^ { H }$ 是相应的偏置。

为了将时间序列恢复到原始尺度，我们使用实例特定的均值和方差进行反归一化。因此，最终预测 $\widehat { \pmb X } \in \mathbb { R } ^ { H \times D }$ 的计算方式为，

$$
{ \widehat { \pmb { X } } } = \operatorname { I n v N o r m } ( f _ { \pmb { \theta } } ( \operatorname { I F F T } ( { \dot { \pmb { X } } } ) ) ) .\tag{6}
$$

## 3.6 优化目标

对于模型优化，我们采用均方误差（MSE）来衡量预测值与真实值之间的损失。受时间序列中数值自相关性 [47] 的启发，我们在频域中引入了平均绝对误差（MAE）损失，以减轻自相关性的影响。总之，我们的优化目标函数 $\mathcal { L }$ 可以表示如下，

$$
{ \mathcal { L } } ( { \widehat { X } } , Y ) = \alpha { \mathrm { M A E } } ( \operatorname { F F T } ( { \widehat { X } } ) , \operatorname { F F T } ( Y ) ) + ( 1 - \alpha ) { \mathrm { M S E } } ( { \widehat { X } } , Y ) ,\tag{7}
$$

其中 $\alpha \in [ 0 , 1 ]$ 是超参数。TimeEmb 的工作流程在附录 B 中有详细说明。

## 3.7 计算效率分析

我们分析了 TimeEmb 核心组件的计算复杂度，即时不变嵌入库和频率过滤。

**时不变嵌入。** 嵌入库 $\pmb { { \cal E } } \in \mathbb { R } ^ { M \times F \times D }$ 支持两种轻量级操作：嵌入查找和频向减法（frequency-wise subtraction）。给定输入时间序列 $\pmb { X } \in \mathbb { R } ^ { L \times D }$，基于其最后一个时间戳检索出嵌入 $\pmb { X } _ { s } \in \mathbb { R } ^ { F \times D }$，复杂度为 $\mathcal { O } ( M )$。减法步骤 $\pmb { X } _ { d } = \overline { \pmb { X } } - \pmb { X } _ { s }$ 涉及 ${ \mathcal { O } } ( F \times D )$ 次操作。因此，整体复杂度是线性的，即 $\mathcal { O } ( M + F \times D )$。

该嵌入库也是参数高效的：例如，在 $L = 9 6 , M = 2 4$、$F = 4 9$ 且 $D = 7$ 的 ETTh1 数据集上，所需的总参数量仅为 $2 4 \times 4 9 \times 7 = 8 , 2 3 2$。

**频率过滤。** 动态分量 $X _ { d }$ 的谱调制是通过与可学习滤波器 $\omega \in \mathbb { C } ^ { F \times 1 }$ 进行逐元素相乘来实现的，其复杂度为 $\mathcal { O } ( F \times D )$。

最后，TimeEmb 中的主要开销源于傅里叶变换，其运算复杂度为 $\mathcal { O } ( D \times$ $L \log L )$。总体而言，关键组件的计算复杂度是线性的，使其对于长序列和多变量输入具有极高的效率和可扩展性。

## 4 实验

在本节中，我们使用真实世界的时间序列基准进行广泛的实验，以充分评估我们所提出模型的性能，包括与最先进（SOTA）基线方法的对比（第 4.2 节）、兼容性评估（第 4.3 节）、时间序列解耦能力分析（第 4.4 节）以及各模块的有效性验证（第 4.5 节）。

## 4.1 实验设置

## 4.1.1 数据集与基线方法

遵循现有时间序列预测研究中主流的评估设置 [51, 58]，我们在七个真实世界的基准数据集上进行实验，包括四个 ETT 数据集（ETTh1、ETTh2、ETTm1、ETTm2）[58]、Weather [51]、Electricity (ECL) [51] 和 Traffic [51]。参考先前的研究工作 [51, 22]，我们将 ETTs 数据集划分训练集、验证集和测试集，比例为 6:2:2，而其他数据集的划分比例为 7:1:2。

为了全面评估有效性，我们从三种代表性框架中选择了全面的 SOTA 基线方法：（1）基于频率的模型：FilterNet [54]、FITS [53] 和 FreTS [55]；（2）基于 MLP 的模型：DLinear [56] 和 CycleNet [14]；以及（3）基于 Transformer 的模型：iTransformer [22]、PatchTST [28] 和 Fredformer [34]。关于数据集和基线方法的详细介绍可见附录 C。

## 4.1.2 实现细节

为了确保公平对比，我们采用通用的实验设置：对所有数据集上的所有基线方法，其回顾窗口长度 $L \in$ {96, 336, 720}，预测长度 $H \in \{ 9 6 , \dot { 1 } 9 2 , 3 3 6 , 7 2 0 \}$ } [51,

表 1：在预测长度 H ∈ {96, 192, 336, 720} 且回顾窗口长度 $L = 9 6 .$ 情况下的性能对比。最佳结果以粗体突出显示，第二佳结果以下划线标出。



<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">TimeEmb (本文模型)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">CycleNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">Fredformer</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">FilterNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">iTransformer</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">PatchTST</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">FITS</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">FreTS</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">DLinear</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.378</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.391</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.373</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.405</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.406</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.395</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.400</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.426</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.419</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.464</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.437</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.476</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.443</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.491</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.462</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.478</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.444</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.499</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.481</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.459</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459±0.002</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.460±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.461</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.460</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.467</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.456</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.474</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.469</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.503</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.491</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.479</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.502</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.495</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.558</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.532</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.519</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.516</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.432</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.426</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.446</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.451</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.475</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.456</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.277±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.328±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.285</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.335</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.297</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.349</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.333</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.356±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.379±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.371</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.389</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.369</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.395</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.380</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.400</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.395</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.395</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.425</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.477</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.476</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.400±0.002</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.417±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.382</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.409</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.428</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.451</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.438</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.462</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.467</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.594</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.541</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.416±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.437±0.002</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.415</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.434</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.446</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.445</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.446</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.721</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.604</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.831</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.657</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.362±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.390±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.404</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.365</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.393</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.378</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.404</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.410</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.408</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.559</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.515</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.304±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.343±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.319</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.360</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.326</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.361</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.318</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.358</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.334</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.368</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.329</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.365</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.355</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.335</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.345</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.354±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.373±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.360</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.363</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.380</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.380</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.380</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.379±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.393±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.389</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.403</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.395</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.403</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.406</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.400</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.410</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.413</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.413</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.435±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.428±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.447</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.438</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.456</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.444</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.491</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.459</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.475</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.486</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.474</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.368±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.384±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.379</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.395</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.410</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.406</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.415</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.408</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.408</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.416</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.403</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.246</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.177</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.174</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.257</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.180</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.184</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.183</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.189</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.277</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.226±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.229</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.290</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.301</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.300</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.246</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.305</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.258</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.326</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.362</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.286±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.284</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.327</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.297</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.348</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.308</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.369</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.389</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.391</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.397</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.412</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.409</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.402</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.399</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.495</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.480</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.554</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.522</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.266</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.314</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.322</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.332</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.330</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.328</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.321</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.368</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.158</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.203</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.163</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.207</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.162</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.207</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.174</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.214</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.176</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.217</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.174</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.208</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.196</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.255</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.200±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.207</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.247</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.211</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.210</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.221</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.254</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.221</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.254</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.219</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.262</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.273</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.335</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.339±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.336±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.341</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.358</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.352</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.334</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.332</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.345</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.243</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.246</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.258</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.270</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.231±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.229</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.147</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.147</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.148</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.176</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.258</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.153±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.246±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.152</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.244</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.165</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.258</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.160</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.162</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.253</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.280</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.175</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.196</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.285</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.177</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.273</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.173</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.267</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.178</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.190</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.214</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.185</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.301</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.299</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.304</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.210</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.225</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.313</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.255</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.327</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.220</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.315</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.333</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167±0.001</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.260±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.168</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.175</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.178</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.270</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.189</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.217</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.189</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.300</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">交通 (Traffic)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432±0.002</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279±0.001</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.406</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.277</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.395</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.268</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.272</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.651</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.593</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.378</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.650</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.442±0.001</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289±0.001</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.457</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.426</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.276</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.602</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.363</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.595</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.598</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.370</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.456±0.002</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295±0.002</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.432</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.281</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.316</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.433</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.450</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.282</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.609</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.366</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.609</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.605</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487±0.003</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311±0.001</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.502</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.314</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.463</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.300</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.498</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.323</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.467</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.484</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.301</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.647</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.673</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.645</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454±0.002</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293±0.001</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.301</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.431</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.310</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.428</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.286</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.627</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.618</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.625</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
    </tr>
  </tbody>
</table>



表 2：回看长度 $L \in$ {336, 720} 下，平均预测长度的性能对比。最佳结果用**粗体**标出，次佳结果用<u>下划线</u>标出。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">历史窗口</th>
      <th colspan="8" style="border: 1px solid #ddd; padding: 8px;">L = 336</th>
      <th colspan="8" style="border: 1px solid #ddd; padding: 8px;">L = 720</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">TimeEmb (本文模型)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">CycleNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">FilterNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">iTransformer</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">TimeEmb (本文模型)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">CycleNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">SOFTS</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">DLinear</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">指标</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.410</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.423</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.415</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.426</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.418</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.433</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.430</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.439</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.455</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm1</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.340</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.371</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.355</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.379</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.352</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.365</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.345</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.376</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.355</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.381</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.247</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.303</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.251</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.309</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.325</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.337</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.248</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.249</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.312</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.331</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.327</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.221</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.255</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.226</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.224</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.239</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.236</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.218</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.257</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.224</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.266</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
    </tr>
  </tbody>
</table>



58, 14]。预测指标包括 MSE 和 MAE，结果为五次独立运行的平均值。TimeEmb 训练 30 个 epoch，并采用早停策略（验证集上的 patience = 5）。ETTs 和 Weather 数据集的 batch size 设为 256，其他数据集设为 64。学习率从 {0.0005, 0.001, 0.002, 0.005} 中选择，TimeEmb 的隐藏层大小固定为 512。实验在 NVIDIA RTX 4090 24GB GPU 上使用 PyTorch 2.1 [33] 进行，详细信息见附录 C。

## 4.2 整体性能

表 1 展示了 $L = 9 6$ 且 $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ 时的对比结果。基准结果来自原论文。可以得出以下几个结论：

(1) TimeEmb 在不同数据集上始终优于强大的基线模型。在多个基准测试和预测步长上，TimeEmb 显著降低了 MSE，平均相对提升范围为 3.0% 至 8.7%。这突出了我们基于频率的动静态分解框架的有效性，该框架显式地分离并建模了时不变（time-invariant）和时变（time-varying）分量。

(2) TimeEmb 通过提供更具表达力和灵活性的分解，超越了基于解耦的基线模型。虽然 CycleNet 依赖于单个长周期嵌入，且 DLinear 采用局部移动平均来进行趋势提取，但这两种方法都难以有效地捕捉长期时间模式。相比之下，TimeEmb 利用全局且感知时间戳的嵌入库（timestamp-aware embedding bank）来学习和表示循环出现的时不变模式，从而实现更准确的长期预测。

表 3：在 Electricity 和 Weather 数据集上将 TimeEmb 与不同骨干网络集成的性能。最佳结果以粗体表示。Impr. 表示装备 TimeEmb 后的性能提升。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th colspan="8" style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</th>
      <th colspan="8" style="border: 1px solid #ddd; padding: 8px;">天气 (Weather)</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">96</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">192</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">336</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">720</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">96</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">192</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">336</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">720</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">指标</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">Linear</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.196</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.195</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.208</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.298</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.330</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.238</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.285</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.335</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.346</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">+ 本文模型</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.173</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.270</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.179</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.274</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.193</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.288</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.233</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.320</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.218</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.222</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.275</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.298</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.349</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.345</strong></td>
    </tr>
    <tr style="border-bottom: 2px solid #ddd; background-color: #fafafa;">
      <td style="border: 1px solid #ddd; font-style: italic; padding: 8px;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+11.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+8.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.4%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+4.1%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+13.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+14.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+11.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.5%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+11.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">-0.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+9.4%</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">MLP</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.177</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.183</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.320</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.180</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.223</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.274</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.342</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.370</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">+ 本文模型</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.137</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.234</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.155</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.250</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.172</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.267</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.211</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.303</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.154</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.197</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.203</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.243</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.263</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.288</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.344</strong></td>
    </tr>
    <tr style="border-bottom: 2px solid #ddd; background-color: #fafafa;">
      <td style="border: 1px solid #ddd; font-style: italic; padding: 8px;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+22.6%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+11.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+15.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+12.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+9.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+5.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+14.4%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+15.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+9.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+11.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+1.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">-0.6%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.0%</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">DLinear</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.195</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.194</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.281</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.207</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.297</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.330</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.195</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.254</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.329</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">+ 本文模型</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.171</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.271</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.181</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.291</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.223</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.321</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.168</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.230</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.216</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.277</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.316</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.333</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.370</strong></td>
    </tr>
    <tr style="border-bottom: 2px solid #ddd; background-color: #fafafa;">
      <td style="border: 1px solid #ddd; font-style: italic; padding: 8px;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+12.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.5%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+0.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+8.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+8.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.7%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+13.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+9.4%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+8.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.1%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+4.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+4.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.9%</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">iTransformer</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.153</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.256</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.182</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.274</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.218</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.181</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.222</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.226</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.260</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.360</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.352</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">+ 本文模型</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.142</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.260</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.175</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.203</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.299</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.162</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.210</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.251</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.269</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.296</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.346</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.344</strong></td>
    </tr>
    <tr style="border-bottom: 2px solid #ddd; background-color: #fafafa;">
      <td style="border: 1px solid #ddd; font-style: italic; padding: 8px;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+1.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+1.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">-1.6%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">-0.4%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+10.5%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+6.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+7.1%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.5%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+5.3%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.0%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+3.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">+2.3%</td>
    </tr>
  </tbody>
</table>



![](images/4c0fb1d40fc63ad98fd82fdd0ec65236b0d6f1c62a02cca6bc405b1ced0920b0.jpg)  
(a) 时间序列 $\overline { { \boldsymbol { X } } }$ 频谱（彩色线）和时不变嵌入 X<sub>s</sub> 频谱（粗红线）

![](images/d5befdd8e6a4f088d316c9728f534ec41fb9cd111d8e56e07ab700e351fd8505.jpg)  
(b) 时变分量 $X _ { d }$ 频谱  
图 3：频域中的解耦特征可视化。右上角放大了 20 到 49 的频率分量。

(3) TimeEmb 通过联合建模时不变和动态分量，优于频域模型。FilterNet 和 FITS 采用了固定的滤波方法，由于忽略了全局时不变模式，这可能无法有效处理非平稳频率分量。相反，除了针对时变分量的频率滤波器外，TimeEmb 中的嵌入还可以保留长期的时不变模式，从而指示时间序列的结构信息。

为了评估模型效率，我们将 Electricity 数据集上的可训练参数量、训练时间和 MSE 与主流基线模型进行了对比，如图 1 所示。值得注意的是，TimeEmb 使用的参数量比代表性的基于 Transformer 的模型 iTransformer 少 5 倍以上，同时实现了最佳的预测性能。得益于其轻量化设计，TimeEmb 在不损失准确性的情况下显著加快了训练速度，展示了效率与效果之间的卓越平衡。

为了进一步评估模型捕获长期依赖关系的能力，我们在延长的回看窗口下评估了 TimeEmb。表 2 报告了 $L = 3 3 6$ 和 $L = 7 2 0$ 时所有预测长度的平均性能。完整结果已推迟至附录 D。TimeEmb 在长输入步长下仍能保持最先进的性能，展示了其强大的时间建模能力。

## 4.3 兼容性分析

为了评估我们提出的用于解耦时不变分量和时变分量的解耦机制的泛化能力，我们将其集成到几种最先进的时间序列预测模型中，涵盖了基于 MLP 和基于 Transformer 的架构。具体而言，我们的方法仅在时域中将骨干预测层替换为其他模型，从而实现了与不同模型的无缝集成。如表 3 所示，引入我们的方法持续提高了各种预测步长下的基线性能，验证了其作为多种预测框架的即插即用增强方案的有效性。重要的是，这种集成带来了极小的计算开销，可在不显著增加模型复杂度或训练成本的情况下实现无缝采用。这些结果突出了我们解耦框架的广泛适用性，以及在几乎没有折中权衡的情况下强化现有模型的潜力。

![](images/6c875fc7f54a99697117ded6878dafe2818fe189aeaeb954c4c8dc9b1d1d17af.jpg)  
(a) 幅值分析

![](images/89bbab725c399a3fb55e42eb5a49b393c7337b2e8bcffd419d34bfb04785c951.jpg)  
(b) 频率分析

![](images/d2f305cfffce55e05e1d5f498d1413e05a9849aa527a858749d29239d7ea917e.jpg)  
(c) ETTh1 上的 MSE

![](images/6147d0c5fb3d66e107b83bdebc7744c5fceda4f2b4f996cb5f327ad6e5f18e79.jpg)  
(d) ETTh1 上的 MAE  
图 5：关键组件贡献分析。

## 4.4 解耦特征可视化

为了评估 TimeEmb 在分离时不变和时变分量方面的解耦能力，我们进行了多种可视化说明。我们首先从 ETTm2 数据集中选择第一个通道，并提取了十几个结束于 0 点（即时间索引 $t _ { l a s t } = 0 )$ 的来自不同日期的时间序列。在图 3(a) 中，我们将这些序列的频域表示 $\overline { { \boldsymbol { X } } }$ 绘制为彩色曲线，并将相应的学习得到的时不变嵌入 $X _ { s }$ 绘制为粗红线。为了清晰起见，我们放大了 20 到 49 范围内的频率分量。我们可以观察到，来自不同日期的多个 X 表现出相似的频谱结构，并且学习到的相应时不变嵌入 $X _ { s }$ 在一定程度上捕捉到了这一共同模式。图 3(b) 展示了时变分量 $X _ { d } ,$ 它们彼此之间相对不同。结果清楚地表明，原始时间序列很难区分，但在减去时不变嵌入后，它们变得更加可分。这表明我们的 TimeEmb 成功捕捉到了输入序列中共享的时不变分量，保留了总体结构信息。

此外，我们从高层视角展示了解耦前后数据的分布。

我们使用 T-SNE [45] 将 Electricity 测试集中的数据样本投影到二维空间中。为了捕捉星期级别的时不变模式，我们添加了一个由 7 个可学习嵌入组成的嵌入库。

![](images/b3157cf7729688c443c0157b2a53bd9c02eb6c8c433503bd0aa012694fdeadf9.jpg)

时间序列 $\overline { { \boldsymbol { X } } }$ 和时变分量 $X _ { d }$ 分别描绘在图 4 (a) 和 (b) 中。对应于一周中不同天的时间序列通过颜色进行编码。例如，周一的时间序列用深蓝色表示，周日的时间序列用浅黄色表示。如图 4 (a) 所示，一周中不同天的时间序列 X 倾向于混杂在一起，这表明它们之间存在固有的相似模式。在从 X 中分离出时不变分量后，一周中不同天的时变分量 $X _ { d }$ 携带了特定信息，如图 4 (b) 中它们清晰且孤立的分布所证实的那样。这种可视化再次支持了我们之前的发现：通过解耦时不变分量 ${ \bar { \mathbf { X } } } _ { s } ,$ 与原始时间序列 $\overline { { \boldsymbol { X } } }$ 相比，时变分量 $X _ { d }$ 变得更加易于区分。

更多关于 TimeEmb 的可视化结果可以在附录 D 中找到。

![](images/241234f3b40ff6eb6974ba951f250a8da1194330abcbbcfbd6390e2fb7f0da7d.jpg)  
图 4：T-SNE 可视化结果。

## 4.5 消融实验

我们进行了全面的消融实验，以评估 TimeEmb 中关键组件的贡献。分析主要从两个角度进行：（1）时不变嵌入的频率组成，以及（2）移除或更改单个模块的影响。

## 4.5.1 $X _ { s }$ 的频谱分析

为了研究不同频率分量对时不变嵌入 $X _ { s }$ 的贡献，我们设计了两种对照扰动策略。基于幅值的掩蔽（Amplitude-based masking）：对于每个输入 X，我们仅保留 X<sub>s</sub> 中幅值最大的前 k 个频率分量，并将其余分量清零。基于频率的滤波（Frequency-based filtering）：我们通过仅保留一定比例的低频分量并丢弃高频部分来应用低通滤波器。结果如图 5 (a) 和 (b) 所示，完整细节见附录 D。如图 5(a) 所示，随着保留更多高幅值分量，模型性能有所提高，这表明主要和次要频率都携带了有用的不变信息 [52]。类似地，图 5(b) 显示增加低频分量的比例会带来更好的性能，反映了在不变表示中捕获短期和长期周期性的重要性。这些发现支持了在构建 $X _ { s }$ 时使用全频谱的做法。

## 4.5.2 组件级消融

为了评估各个关键模块的独立影响，我们通过更改或移除组件构建了 TimeEmb 的几种变体：Random：在训练和测试之间，嵌入库是随机初始化的。Zero/Mean：嵌入库分别固定为全零或全局均值。w/o $X _ { s } \colon$ 时不变分量被完全移除。w/o $\mathcal { H } _ { \omega } \mathrm { . }$：频率滤波器从动态处理路径中被移除。图 5 (c) 和 (d) 中的结果表明，嵌入库和频率滤波器都对模型性能有实质性的贡献。特别是，移除其中任何一个模块都会导致性能显著下降，这证实了联合建模时不变分量和时变分量的重要性。完整的消融结果在附录 D 中报告。总的来说，这些发现验证了我们系统性解耦框架的有效性，其中 $X _ { s }$ 和 $X _ { d }$ 通过专用结构进行独立处理，以捕获互补的时间特征。

## 5 结论

在本文中，我们使用一个结构良好的分解框架来解决时间序列预测中时间非平稳性的关键问题。我们引入了 TimeEmb，这是一种将全局时间嵌入和频谱滤波相结合的轻量且有效的架构。TimeEmb 能够对解耦的时变分量和时不变分量进行单独处理。具体来说，我们利用可学习的嵌入来保留时间序列内的长期不变模式。此外，我们设计了一个频率滤波器来捕捉时变分量的时间依赖性。广泛的实验证实，我们的方法不仅达到了最先进的性能，而且通过其双路径设计为时间模式提供了可解释的见解。它在性能和效率之间实现了出色的平衡。此外，它可以轻松地与现有方法集成，从而提高预测时间序列的能力。

## 致谢

本研究工作得到了中国博士后科学基金项目（资助号 2025M771587）、国家科技重大专项（资助号 2021ZD0112500）、国家自然科学基金项目（资助号 U22A2098, 62172185, 62206105 和 62202200）、吉林省科技发展计划重大项目（资助号 20240212003GX）以及长春市科技发展计划重大项目（资助号 2024WX05）的资助。

## 参考文献

[1] Richard Asselin. Frequency filter for time integrations. Monthly Weather Review, 100(6): 487–490, 1972.

[2] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, and Greta M Ljung. Time series analysis: forecasting and control. John Wiley & Sons, 2015.

[3] RN Bracewell. Signal analysis. Proceedings of the IEEE, 66(9):1101–1102, 1978.

[4] Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O Arik, and Tomas Pfister. Tsmixer: An all-mlp architecture for time series forecasting. TMLR, 2023.

[5] James W Cooley and John W Tukey. An algorithm for the machine calculation of complex fourier series. Mathematics of computation, 19(90):297–301, 1965.

[6] Tao Dai, Beiliang Wu, Peiyuan Liu, Naiqi Li, Jigang Bao, Yong Jiang, and Shu-Tao Xia. Periodicity decoupling framework for long-term series forecasting. In The International Conference on Learning Representations, 2024.

[7] Jie Feng, Yong Li, Chao Zhang, Funing Sun, Fanchao Meng, Ang Guo, and Depeng Jin. Deepmove: Predicting human mobility with attentional recurrent networks. In WWW, pages 1459–1468, 2018.

[8] Jean-Yves Franceschi, Aymeric Dieuleveut, and Martin Jaggi. Unsupervised scalable representation learning for multivariate time series. In NIPS, volume 32, pages 4650–4661. Curran Associates, Inc., 2019.

[9] Hannah Rosa Friesacher, Emma Svensson, Susanne Winiwarter, Lewis Mervin, Adam Arany, and Ola Engkvist. Temporal distribution shift in real-world pharmaceutical data: Implications for uncertainty quantification in qsar models. arXiv preprint arXiv:2502.03982, 2025.

[10] Lu Han, Xu-Yang Chen, Han-Jia Ye, and De-Chuan Zhan. Softs: Efficient multivariate time series forecasting with series-core fusion. In NeurIPS, 2024.

[11] Tao Hong, Pierre Pinson, Yi Wang, Rafał Weron, Dazhi Yang, and Hamidreza Zareipour. Energy forecasting: A review and outlook. IEEE Open Access Journal of Power and Energy, 7:376–388, 2020.

[12] Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In ICLR, 2019.

[13] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In SIGIR, pages 95–104, 2018.

[14] Shengsheng Lin, Weiwei Lin, Xinyi Hu, Wentai Wu, Ruichao Mo, and Haocheng Zhong. Cyclenet: Enhancing time series forecasting through modeling periodic patterns. In NeurIPS, 2024.

[15] Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, and Rui Zhao. Timecma: Towards llm-empowered multivariate time series forecasting via cross-modality alignment. In AAAI, 2025.

[16] Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, and Rui Zhao. Timecma: Towards llm-empowered multivariate time series forecasting via cross-modality alignment. In AAAI, volume 39, pages 18780–18788, 2025.

[17] Chenxi Liu, Shaowen Zhou, Qianxiong Xu, Hao Miao, Cheng Long, Ziyue Li, and Rui Zhao. Towards cross-modality modeling for time series analytics: A survey in the llm era. In IJCAI, 2025.

[18] Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, and Qiang Xu. Scinet: time series modeling and forecasting with sample convolution and interaction. In NIPS, pages 5816–5828, 2022.

[19] Peiyuan Liu, Hang Guo, Tao Dai, Naiqi Li, Jigang Bao, Xudong Ren, Yong Jiang, and Shu-Tao Xia. Calf: Aligning llms for time series forecasting via cross-modal fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18915–18923, 2025.

[20] Peiyuan Liu, Beiliang Wu, Yifan Hu, Naiqi Li, Tao Dai, Jigang Bao, and Shu-tao Xia. Timebridge: Non-stationarity matters for long-term time series forecasting. 2025.

[21] Yong Liu, Chenyu Li, Jianmin Wang, and Mingsheng Long. Koopa: Learning non-stationary time series dynamics with koopman predictors. NeurIPS, 36:12271–12290, 2023.

[22] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In ICLR, 2024.

[23] R. T. M. C. Lu. Algorithms for Discrete Fourier Transform and Convolution. Springer, Berlin, Heidelberg, 1989. ISBN 978-1-4419-5126-4.

[24] Yucong Luo, Yitong Zhou, Mingyue Cheng, Jiahao Wang, Daoyu Wang, Tingyue Pan, and Jintao Zhang. Time series forecasting as reasoning: A slow-thinking approach with reinforced llms. arXiv preprint arXiv:2506.10630, 2025.

[25] Hao Miao, Ziqiao Liu, Yan Zhao, Chenjuan Guo, Bin Yang, Kai Zheng, and Christian S Jensen. Less is more: Efficient time series dataset condensation via two-fold modal matching. PVLDB, 18(2):226–238, 2024.

[26] Hao Miao, Yan Zhao, Chenjuan Guo, Bin Yang, Kai Zheng, Feiteng Huang, Jiandong Xie, and Christian S. Jensen. A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data. In ICDE, pages 1050–1062, 2024.

[27] RE Morrow. The fast fourier transform. IEEE spectrum, 4(12):63–70, 1967.

[28] Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In ICLR, 2023.

[29] Kin G. Olivares, Cristian Challú, Azul Garza, Max Mergenthaler Canseco, and Artur Dubrawski. NeuralForecast: User friendly state-of-the-art neural forecasting models. PyCon Salt Lake City, Utah, US 2022, 2022. URL https://github.com/Nixtla/neuralforecast.

[30] Alan V Oppenheim. Discrete-time signal processing. Pearson Education India, 1999.

[31] Boris N. Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. In ICLR, 2020.

[32] Pranay Pasula. Real world time series benchmark datasets with distribution shifts: Global crude oil price and volatility. arXiv preprint arXiv:2308.10846, 2023.

[33] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, pages 8024–8035, 2019.

[34] Xihao Piao, Zheng Chen, Taichi Murayama, Yasuko Matsubara, and Yasushi Sakurai. Fredformer: Frequency debiased transformer for time series forecasting. In SIGKDD, pages 2400–2410, 2024.

[35] Syama Sundar Rangapuram, Matthias Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. In NIPS, pages 7796–7805, 2018.

[36] Syama Sundar Rangapuram, Matthias W Seeger, Jan Gasthaus, Lorenzo Stella, Yuyang Wang, and Tim Januschowski. Deep state space models for time series forecasting. 31, 2018.

[37] CLEVELAND RB. Stl: A seasonal-trend decomposition procedure based on loess. J Off Stat, 6:3–73, 1990.

[38] Weijieying Ren, Tianxiang Zhao, Wei Qin, and Kunpeng Liu. T-sas: Toward shift-aware dynamic adaptation for streaming data. In CIKM, pages 4244–4248, 2023.

[39] Zezhi Shao, Zhao Zhang, Fei Wang, Wei Wei, and Yongjun Xu. Spatial-temporal identity: A simple yet effective baseline for multivariate time series forecasting. In CIKM, page 4454–4458, 2022.

[40] Zezhi Shao, Zhao Zhang, Wei Wei, Fei Wang, Yongjun Xu, Xin Cao, and Christian S Jensen. Decoupled dynamic spatial-temporal graph neural network for traffic forecasting. PVLDB, 15 (11):2733–2746, 2022.

[41] Zezhi Shao, Zhao Zhang, Wei Wei, Fei Wang, Yongjun Xu, Xin Cao, and Christian S Jensen. Decoupled dynamic spatial-temporal graph neural network for traffic forecasting. Proceedings of the VLDB Endowment, 15(11):2733–2746, 2022.

[42] Pushpendra Singh, Shiv Dutt Joshi, Rakesh Kumar Patney, and Kaushik Saha. The fourier decomposition method for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2199):20160871, 2017.

[43] Yi Tay, Mostafa Dehghani, Jinfeng Rao, William Fedus, Samira Abnar, Hyung Won Chung, Sharan Narang, Dani Yogatama, Ashish Vaswani, and Donald Metzler. Scale efficiently: Insights from pretraining and finetuning transformers. In ICLR, 2020.

[44] Stephen J Taylor. Modelling financial time series. world scientific, 2008.

[45] Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. JMLR, 9(86): 2579–2605, 2008.

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, volume 30, 2017.

[47] Hao Wang, Lichen Pan, Yuan Shen, Zhichao Chen, Degui Yang, Yifei Yang, Sen Zhang, Xinggao Liu, Haoxuan Li, and Dacheng Tao. Label correlation biases direct time series forecast. In ICLR, 2025.

[48] Jiahao Wang, Mingyue Cheng, and Qi Liu. Can slow-thinking llms reason over time? empirical studies in time series forecasting. arXiv preprint arXiv:2505.24511, 2025.

[49] Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang, and JUN ZHOU. Timemixer: Decomposable multiscale mixing for time series forecasting. In ICLR, 2024.

[50] Jan C Willems. From time series to linear system—part i. finite dimensional linear time invariant systems. Automatica, 22(5):561–580, 1986.

[51] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In NeurIPS, volume 34, pages 22419–22430, 2021.

[52] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In ICLR, 2023.

[53] Zhijian Xu, Ailing Zeng, and Qiang Xu. Fits: Modeling time series with 10k parameters. In ICLR, 2024.

[54] K. Yi, J. Fei, Q. Zhang, H. He, S. Hao, D. Lian, and W. Fan. Filternet: Harnessing frequency filters for time series forecasting. In NeurIPS, 2024.

[55] Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, and Zhendong Niu. Frequency-domain MLPs are more effective learners in time series forecasting. In NeurIPS, 2023.

[56] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In AAAI, volume 37, pages 11121–11128, 2023.

[57] Junbo Zhang, Yu Zheng, Dekang Qi, Ruiyuan Li, and Xiuwen Yi. Dnn-based prediction model for spatio-temporal data. In Proceedings of the 24th ACM SIGSPATIAL international conference on advances in geographic information systems, pages 1–4, 2016.

[58] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In AAAI, volume 35, pages 11106–11115, 2021.

[59] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In ICML, pages 27268–27286, 2022.

## A TimeEmb的深入分析

## A.1 创新性讨论

TimeEmb 与解耦方法。 虽然先前的解耦方法 [31, 51] 主要侧重于根据单个时间序列内的局部统计数据来分离趋势和残差分量，但 TimeEmb 引入了两个根本性的进展。首先，我们的模型不是进行局部解耦，而是利用可学习的嵌入库（embedding bank）来捕获整个数据集中全局一致且循环出现的模式，从而有效地保留系统级不变量。其次，TimeEmb 首次将可学习的频域滤波引入解耦框架中，从而能够在谱空间中对动态分量进行高效且具有强表达能力的建模。

TimeEmb 与嵌入增强方法。 嵌入增强模型 [40, 39] 通常使用基于标识符的嵌入 $( e . g .$ , time slot, spatial ID) 来编码辅助信息。相比之下，TimeEmb 采用了基于分解的设计，其中可学习的时间嵌入库显式地对时间不变的信号分量进行建模。这使得模型能够以数据驱动的方式恢复潜在的周期性模式，而无需依赖预定义的标识符或外部先验。

## A.2 理论支持

在本节中，我们从频域的角度对 TimeEmb 的核心设计进行理论分析。我们主要关注两个方面：频域表示与操作的完整性，以及用于建模动态时间信号的可学习谱滤波机制的表达能力。

## A.2.1 频域表示与操作的完整性

TimeEmb 完全在频域中运行，它对输入序列应用实值快速傅里叶变换（rFFT）。对于实值时间序列 $\pmb { X } ^ { ^ {  } } \in \mathbb { R } ^ { L \times D }$ ，其谱表示形式为 $\overline { { \boldsymbol { X } } } \in \mathbb { C } ^ { F \times D }$ ，其中由于频谱的共轭对称性， $F = \lfloor L / 2 \rfloor + 1$ 。rFFT 定义为：

$$
{ \overline { { \mathbf { X } } } } [ k ] = \sum _ { n = 0 } ^ { L - 1 } { X [ n ] \cdot e ^ { - 2 \pi j k n / L } } , \quad k = 0 , \ldots , F - 1 .\tag{8}
$$

该变换可以通过相应的逆实值 FFT（irFFT）进行逆变换，从而保证在此过程中不会丢失任何信息。因此，rFFT 为实值信号提供了完整且高效的频率表示 [30, 5]。除了变换之外，TimeEmb 还完全在频域中执行了一系列操作：

1. 从输入频谱 $\overline { { \boldsymbol { X } } }$ 中减去时间不变的嵌入 $X _ { s }$

2. 通过可学习的滤波器 $\omega ;$ 对残差 $\pmb { X } _ { d } = \overline { { \pmb { X } } } - \pmb { X } _ { s }$ 进行逐频率调制；

3. 重构最终的频谱 $\dot { \boldsymbol { X } } = \boldsymbol { X } _ { s } + \boldsymbol { X } _ { d } \odot \boldsymbol { \omega }$ ，然后进行 irFFT 以在时域中恢复输出。

这些操作（即 $i . e . ,$ 减法、调制和加法）中的每一个在代数上都是定义良好且在频域中是封闭的。因为 rFFT 是可逆的，所以 TimeEmb 中的整个变换链是表示完整的：保留了所有原始信息，同时允许在谱空间中进行结构化操作。

该设计具有几个重要的优点。首先，它能够对周期性和振荡行为进行精确建模，而这些行为在时域中往往难以定位。其次，完全在频空间中工作可以对长程时间模式进行高效且可解释的分解。最后，该模型避免了由于投影或截断而导致的任何信息丢失，从而确保了其设计在理论上的合理性。

## A.2.2 频域滤波的表达能力

为了对输入序列的动态（随时间变化的）分量进行建模，TimeEmb 在残差频谱上应用频域滤波器。形式上，给定残差 $\pmb { X } _ { d } \in \mathbb { C } ^ { F \times D }$ ，可学习的

调制向量 $\boldsymbol{\omega} \in \mathbb { C } ^ { F \times 1 }$ 的应用方式如下：

$$
\begin{array} { r } { \mathcal { H } _ { \omega } ( X _ { d } ) [ k ] = X _ { d } [ k ] \odot \omega [ k ] . } \end{array}\tag{9}
$$

该操作基于卷积定理：频域中的点乘对应于时域中的卷积 [30]。因此，频域滤波器可以被解释为直接在谱域中学习线性时不变（LTI）系统的冲激响应。

这种解释赋予了模型若干表达和实用上的优势 [3]。首先，它能够学习超出局部卷积的灵活信号变换，例如 $e . g .$ ，通过全局频率感知操作捕获长程依赖。其次，滤波过程的计算效率很高，运行复杂度为 $\mathcal { O } ( F \times D )$ ，并且避免了时域 CNN 中固有的卷积核长度限制。最后，该公式对模型对各种周期性结构的敏感性提供了直观的控制，允许其根据特定任务的动态特性来强调或抑制频谱带。

本质上，TimeEmb 中的频域滤波器作为一个强大且紧凑的算子，能够模拟广泛的频谱响应族。

## A.3 卷积定理

频域滤波会改变信号的频率内容。给定信号 $x [ n ]$ 和一个具有频率响应 $H [ k ]$ 的滤波器，滤波后的信号在频域中表示为 $Y [ k ] = X [ k ] H [ k ]$ 。根据卷积定理，滤波后的信号在时域中表示为 $y [ n ] = \mathrm { I D F T } ( Y [ k ] ) = ( x \circledast h ) [ n ]$ 。其证明如下：

令 $x [ n ]$ 和 $h [ n ]$ 为长度为 N 的序列，其 DFT 分别为 $X [ k ]$ 和 $H [ k ]$ ：

$$
X [ k ] = \sum _ { n = 0 } ^ { N - 1 } x [ n ] e ^ { - j { \frac { 2 \pi } { N } } k n } , \quad k = 0 , 1 , \cdots , N - 1 ,\tag{10}
$$

$$
H [ k ] = \sum _ { n = 0 } ^ { N - 1 } h [ n ] e ^ { - j { \frac { 2 \pi } { N } } k n } , \quad k = 0 , 1 , \cdots , N - 1 .\tag{11}
$$

$x [ n ]$ 和 $h [ n ]$ 的循环卷积定义为 $\begin{array} { r } { y [ n ] = ( x \circledast h ) [ n ] = \sum _ { m = 0 } ^ { N - 1 } x [ m ] h [ ( n - } \end{array}$ $m )$ mod $N ]$ ，其中 $\circledast$ 表示循环卷积操作，而 $( n - m )$ mod N 表示 $n - m$ 对 N 的取模操作。

$y [ n ]$ 的 DFT 记为 $Y [ k ]$ ：

$$
\begin{array} { r l r } {  { Y [ k ] = \sum _ { n = 0 } ^ { N - 1 } y [ n ] e ^ { - j \frac { 2 \pi } { N } k n } } } \\ & { } & { = \sum _ { n = 0 } ^ { N - 1 } ( \sum _ { m = 0 } ^ { N - 1 } x [ m ] h [ ( n - m ) \bmod N ] ) e ^ { - j \frac { 2 \pi } { N } k n } . } \end{array}\tag{12}
$$

(13)

令 $l = ( n - m )$ mod N，上述方程可以重写为：

$$
Y [ k ] = N \sum _ { m = 0 } ^ { N - 1 } x [ m ] \sum _ { l = 0 } ^ { N - 1 } h [ l ] e ^ { - j { \frac { 2 \pi } { N } } k ( l + m ) } .\tag{14}
$$

根据指数运算规则，我们有：

$$
Y [ k ] = \left( \sum _ { m = 0 } ^ { N - 1 } x [ m ] e ^ { - j { \frac { 2 \pi } { N } } k m } \right) \left( \sum _ { l = 0 } ^ { N - 1 } h [ l ] e ^ { - j { \frac { 2 \pi } { N } } k l } \right) .\tag{15}
$$

因此：

$$
Y [ k ] = X [ k ] H [ k ] .\tag{16}
$$

算法 1 TimeEmb 的工作流程。   
输入：时间序列 $\overline { { \boldsymbol { X } \in \mathbb { R } ^ { L \times D } } } .$   
输出：预测值 $\widehat { \pmb X } \in \mathbb { R } ^ { H \times D }$   
1: // 域变换   
2: X = FFT(InstNorm(X))   
3: // 时间序列解耦   
4: $\pmb { X } _ { d } = \overline { { \pmb { X } } } - \pmb { X } _ { s }$ {Eq. (2)}   
5: // 频域滤波   
6: $\dot { \boldsymbol X } = \mathcal { H } _ { \omega } ( \boldsymbol X _ { d } ) + \boldsymbol X _ { s }$ <sub>s</sub> {Eq. (4)}   
7: // 最终预测   
8: X = InvNorm(f<sub>θ</sub>(IFFT(X<sup>˙</sup> ))) {Eq. (6)}   
9: 返回：$\widehat { X }$

表 4：数据集统计信息。“Channels” 表示每个数据集中的变量数；“M of each bank” 表示 TimeEmb 中使用的每个嵌入库的容量。“d” 指天级嵌入库，而 “w” 表示周级嵌入库。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.95em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">数据集s</th>
      <th style="border: 1px solid #ddd; padding: 8px;">ETTh1</th>
      <th style="border: 1px solid #ddd; padding: 8px;">ETTh2</th>
      <th style="border: 1px solid #ddd; padding: 8px;">ETTm1</th>
      <th style="border: 1px solid #ddd; padding: 8px;">ETTm2</th>
      <th style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</th>
      <th style="border: 1px solid #ddd; padding: 8px;">天气 (Weather)</th>
      <th style="border: 1px solid #ddd; padding: 8px;">交通 (Traffic)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Channels</td>
      <td style="border: 1px solid #ddd; padding: 8px;">7</td>
      <td style="border: 1px solid #ddd; padding: 8px;">7</td>
      <td style="border: 1px solid #ddd; padding: 8px;">7</td>
      <td style="border: 1px solid #ddd; padding: 8px;">7</td>
      <td style="border: 1px solid #ddd; padding: 8px;">321</td>
      <td style="border: 1px solid #ddd; padding: 8px;">21</td>
      <td style="border: 1px solid #ddd; padding: 8px;">862</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Timesteps</td>
      <td style="border: 1px solid #ddd; padding: 8px;">17420</td>
      <td style="border: 1px solid #ddd; padding: 8px;">17420</td>
      <td style="border: 1px solid #ddd; padding: 8px;">69680</td>
      <td style="border: 1px solid #ddd; padding: 8px;">69680</td>
      <td style="border: 1px solid #ddd; padding: 8px;">26304</td>
      <td style="border: 1px solid #ddd; padding: 8px;">52696</td>
      <td style="border: 1px solid #ddd; padding: 8px;">17544</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Frequency</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Hourly</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Hourly</td>
      <td style="border: 1px solid #ddd; padding: 8px;">15min</td>
      <td style="border: 1px solid #ddd; padding: 8px;">15min</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Hourly</td>
      <td style="border: 1px solid #ddd; padding: 8px;">10min</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Hourly</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">Domain</td>
      <td style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">交通 (Traffic)</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px; text-align: left; font-weight: bold;">M of each bank</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d) + 7 (w)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">24 (d) + 7 (w)</td>
    </tr>
  </tbody>
</table>



逆向推导也是如此。我们最终可以推断出：

$$
y [ n ] = \mathrm { I D F T } ( Y [ k ] ) = ( x \circledast h ) [ n ] = \sum _ { m = 0 } ^ { N - 1 } x [ m ] h [ ( n - m ) \mod N ] .\tag{17}
$$

综上所述，我们已经证明了循环卷积的 DFT 等于 DFT 的乘积，并且 DFT 乘积的 IDFT 等于循环卷积，这意味着频域滤波（DFT 域中的乘法）等价于时域中的循环卷积。

## B 算法

我们在算法 1 中展示了 TimeEmb 的流程。我们首先进行频率分析的域变换（第 2 行）。具体来说，我们对输入序列 X 应用实例归一化，然后进行快速傅里叶变换（FFT）以获得频域序列 $\mathbf { \bar { X } }$ 。接下来，为了解耦时间序列，我们从嵌入库中检索相应的嵌入，该嵌入作为时间不变的分量 $X _ { s } .$ 然后，我们将其从 $\overline { { \boldsymbol { X } } }$ 中分离出来以提取随时间变化的分量 $X _ { d }$ （第 4 行）。随后，我们使用谱调制算子 ω 实现频域滤波，以有效地对动态分量进行建模。在此步骤之后，我们加回时间不变序列 $X _ { s }$ （第 6 行）。然后，组合序列经历快速傅里叶逆变换（IFFT），接着通过一个投影层，最后进行逆归一化以生成最终的预测值（第 8 行）。

## C 实验设置

## C.1 数据集

我们在这里详细描述数据集：

ETT（电力变压器温度，Electricity Transformer Temperature）包含两个数据子集：ETTh 和 ETTm。这些数据集基于每小时和 15 分钟的间隔，收集自 2016 年 7 月至 2018 年 7 月之间的电力变压器。

Weather 记录了 2020 年全年每十分钟的 21 个天气特征，包括气温和湿度。

Electricity 收集了 2012 年至 2014 年 321 个客户的每小时电力消耗。


Traffic 数据集包含自 2015 年至 2016 年旧金山高速公路上 862 个传感器的每小时数据。

详细的统计数据如表 4 所示。

## C.2 基线模型

我们将 TimeEmb 与 9 种具有代表性且最先进的模型进行对比，以评估其性能和有效性，包括基于频率的模型、基于 MLP 的模型以及基于 Transformer 的模型。这些基线模型的详细信息如下：

FilterNet 提出了两种可学习的滤波器——普通整形滤波器（Plain shaping filter）和上下文整形滤波器（Contextual shaping filter），以近似替代时间序列文献中广泛采用的线性映射和注意力映射。其详细实现可在 https://github.com/aikunyi/FilterNet 获取。

FITS 通过在复频域中进行插值来开展时间序列分析，仅需 10K 个参数即可实现低成本计算。其详细实现可在 https://github.com/ VEWOXIC/FITS 获取。

FreTS 提出了一种在频域中利用 MLP 的新方法，能够有效捕捉时间序列的潜在模式，同时受益于全局视角和能量集中。其详细实现可在 https://github.com/aikunyi/FreTS 获取。

DLinear 采用简单的单层线性模型，通过季节-趋势分解来捕捉时间关系。其详细实现可在 https://github.com/ cure-lab/LTSF-Linear 获取。

SOFTS 引入了一种高效的基于 MLP 的模型，该模型采用集中式策略以提升性能，并减轻对各个通道质量的依赖。其详细实现可在 https://github.com/Secilia-Cxy/SOFTS 获取。

CycleNet 利用 RCF 技术来分离序列内固有的周期性模式，然后对所建模周期的残差分量进行预测。其详细实现可在 https://github.com/ACAT-SCUT/CycleNet 获取。

iTransformer 在转置维度上应用注意力机制和前馈网络。它将单条序列的时间点嵌入为变量特征向量（variate tokens），以便注意力机制捕捉多变量相关性。此外，前馈网络被应用于每个变量特征向量以学习非线性表示。其详细实现可在 https://github.com/thuml/ iTransformer 获取。

PatchTST 将时间序列数据分解为子序列级别的 patch，这有助于提取局部语义信息。其详细实现可在 https://github.com/ yuqinie98/PatchTST 获取。

Fredformer 是一个基于 Transformer 的框架，它通过在不同频段上平等地学习特征来解决频率偏差问题，从而确保模型不会忽略对于准确预测至关重要的低振幅特征。其详细实现可在 https: //github.com/chenzRG/Fredformer 获取。

## C.3 实现细节

我们使用 PyTorch 实现了 TimeEmb，并在配备 24GB 显存的单张 NVIDIA RTX4090 GPU 上进行了实验。TimeEmb 训练了 30 个 epoch，并基于验证集实现了早停机制，忍耐度（patience）设为 5。ETT 和 Weather 数据集的批大小（batch size）均设为 256，而其余数据集的批大小则设为 64。这一调整是必要的，因为后者数据集具有更多的通道数，需要较小的批大小以防止显存溢出问题。学习率根据验证集上的表现，从 0.0005, 0.001, 0.002, 0.005 中选择。TimeEmb 中隐藏层的维度一致设置为 512。

表 5：回溯长度 L = 336 的完整结果。最佳结果加粗表示，次佳结果加下划线表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">TimeEmb (本文模型)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">CycleNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">FilterNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">SOFTS</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">iTransformer</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">DLinear</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.367</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.394</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.374</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.396</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.379</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.404</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.405</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.415</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.374</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.403</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.414</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.406</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.415</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.428</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.428</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.438</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.422</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.431</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.430</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.443</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.451</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.442</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.445</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.446</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.450</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.464</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.476</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.476</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.485</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.497</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.507</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.410</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.423</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.415</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.426</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.276</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.333</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.279</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.341</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.356</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.298</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.356</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.334</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.379</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.335</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.378</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.342</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.385</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.360</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.413</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.404</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.370</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.405</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.371</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.413</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.415</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.438</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.396</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.433</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.451</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.414</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.444</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.454</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.598</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.549</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.344</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.355</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.398</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.361</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.402</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.399</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.332</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.348</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.344</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.303</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.357</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.323</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.361</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.334</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.367</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.331</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.369</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.374</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.345</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.353</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.380</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.368</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.386</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.364</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.371</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.397</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.397</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.403</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.410</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.417</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.414</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.425</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.340</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.371</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.355</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.379</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.352</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.359</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.365</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.160</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.243</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.159</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.247</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.177</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.174</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.184</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.273</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.165</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.257</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.218</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.283</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.214</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.286</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.232</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.304</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.322</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.227</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.316</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.269</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.322</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.351</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.304</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.362</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.346</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.370</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.363</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.382</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.402</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.247</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.303</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.251</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.309</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.325</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.326</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.337</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.144</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.189</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.148</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.150</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.183</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.160</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.163</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.174</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.235</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.187</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.233</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.190</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.221</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.204</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.203</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.219</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.271</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.243</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.246</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.258</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.253</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.315</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.326</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.322</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.295</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.335</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.326</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.338</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.363</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.221</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.255</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.226</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.224</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.239</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.270</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.236</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.128</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.223</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.128</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.223</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.132</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.224</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.127</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.221</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.133</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.229</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.140</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.146</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.240</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.144</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.143</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.148</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.156</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.153</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.161</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.160</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.254</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.155</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.253</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.198</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.198</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.287</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.195</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.202</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.304</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.203</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.158</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.252</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.158</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.250</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.156</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.252</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.161</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.254</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">交通 (Traffic)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.346</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.361</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.255</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.410</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.404</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.303</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.373</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.258</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.380</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.268</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.411</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.416</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.312</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.385</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.389</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.273</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.445</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.300</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.325</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.419</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.283</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.415</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.285</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.466</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.315</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.277</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.413</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.263</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.386</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.270</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
    </tr>
  </tbody>
</table>



## D 详细结果

## D.1 回溯窗口长度 L ∈ {336, 720} 的完整结果

为了评估 TimeEmb 在建模长期时间依赖关系方面的性能，我们进一步在延长的回溯窗口长度（336 和 720）下进行了实验。如表 5 和表 6 所示，在这些极具挑战性的设定下，TimeEmb 在各种预测步长（forecast horizons）内均一致取得了具有竞争力的或更优的性能。与许多随着输入长度增加而性能显著下降的基线模型不同，TimeEmb 保持了稳定的准确度，展现出强大的时间泛化能力。

这一性能源于 TimeEmb 的架构设计。时不变嵌入库（time-invariant embedding bank）使模型能够有效总结循环出现的结构模式，而与输入长度无关。同时，频域滤波器自适应地强调相关的动态分量，而不受局部感受野的限制。这些模块协同作用，使 TimeEmb 能够高效地捕捉长程依赖关系和局部变化。

总体而言，这些结果表明 TimeEmb 不仅在标准设定下行之有效，而且在应用于长上下文预测任务时表现出强大的可扩展性与韧性——这对于现实世界的时间序列应用而言是一个非常理想的特性。

表 6：回溯长度 L = 720 的完整结果。最佳结果加粗表示，次佳结果加下划线表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">TimeEmb (本文模型)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">CycleNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">FilterNet</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">SOFTS</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">iTransformer</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">DLinear</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.372</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.400</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.379</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.403</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.416</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.379</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.402</u></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.413</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.427</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.416</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.442</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.419</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.438</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.443</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.445</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.450</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.446</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.468</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.475</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.456</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.456</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.449</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.462</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.477</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.483</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.484</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.488</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.481</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.500</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.525</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.520</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.493</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.506</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.418</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.433</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.430</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.439</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.455</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.457</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.469</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.290</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.348</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.271</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.337</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.297</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.357</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.357</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.369</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.347</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.387</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.332</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.380</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.361</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.400</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.365</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.409</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.409</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.376</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.411</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.362</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.408</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.397</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.403</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.508</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.495</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.399</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.439</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.415</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.449</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.448</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.473</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.851</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.653</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.353</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.396</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.345</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.394</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.412</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.379</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.419</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.519</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.489</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.293</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.346</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.353</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.301</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.358</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.299</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.357</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.353</u></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.326</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.337</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.371</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.379</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.381</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.345</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.356</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.382</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.364</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.387</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.377</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.402</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.405</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.410</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.410</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.411</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.443</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.436</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.345</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.376</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.355</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.381</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.363</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.368</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.396</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.163</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.251</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.159</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.249</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.180</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.181</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.187</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.163</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.219</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.290</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.214</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.289</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.313</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.310</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.319</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.220</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.300</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.320</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.268</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.326</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.341</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.355</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.343</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.372</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.353</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.384</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.361</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.394</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.411</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.406</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.248</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.249</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.312</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.330</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.331</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.341</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.327</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.143</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.193</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.149</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.203</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.153</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.208</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.152</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.205</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.222</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.227</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.188</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.192</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.244</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.199</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.199</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.256</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.236</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.274</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.242</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.283</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.306</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.325</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.312</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.333</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.313</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.333</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.322</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.337</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.352</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.319</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.359</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.218</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.257</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.224</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.266</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.228</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.270</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.129</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.225</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.128</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.223</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.235</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.232</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.142</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.134</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.232</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.145</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.241</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.143</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.160</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.157</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.252</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.160</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.148</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.161</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.257</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.159</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.254</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.174</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.274</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.179</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.163</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.197</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.289</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.197</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.287</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.198</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.220</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.316</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.198</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.158</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.253</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.157</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.250</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.175</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.161</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">交通 (Traffic)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.374</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.374</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.285</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.355</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.253</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.358</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.254</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.401</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.369</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.261</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.375</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.263</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.399</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.401</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.275</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.405</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.408</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.271</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.273</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.292</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.409</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.418</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.292</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.308</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.399</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.274</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.403</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.411</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.380</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.268</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.385</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><u>0.271</u></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.413</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
    </tr>
  </tbody>
</table>



## D.2 效率分析的完整结果

为了验证 TimeEmb 的轻量化特性，我们对其进行了多项效率实验。首先，我们在 ETTm1 数据集上将最大内存占用（MB）、训练时间以及 MSE 与主流基线模型进行了对比。随后，我们基于兼容性研究计算了 TimeEmb 引入的额外参数量，这也能够证明 TimeEmb 的可扩展性与效率。详细结果如表 7 和表 8 所示。结果表明，TimeEmb 可以在资源受限的环境中运行并取得出色的性能。此外，TimeEmb 还可以作为其他模型中的插件模块，以极低的成本提升其性能。

## D.3 消融实验的完整结果

## D.3.1 嵌入频谱分析的消融结果

我们对嵌入频谱进行消融实验的完整结果展示在表 9 和表 10 中。表 9 中的术语“k”代表根据振幅前几位选取的 $X _ { s }$ 频率分量数量，而表 10 中的术语 $" \gamma "$ 则是指应用于嵌入的低通滤波比例。结果表明，利用整个频谱能带来最佳的性能。值得注意的是，覆盖的频率分量越多，TimeEmb 所能取得的性能就越好，这表明该频段内的每个频率都至关重要。

表 7：在 ETTm1 数据集上的效率对比表明 TimeEmb 在性能和效率上均处于领先地位。


<table style="border-collapse: collapse; width: 60%; border: 1px solid #ddd; text-align: center; margin: 0 auto; font-size: 0.95em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 8px;">模型</th>
      <th style="border: 1px solid #ddd; padding: 8px;">训练时间 (秒/轮)</th>
      <th style="border: 1px solid #ddd; padding: 8px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 8px;">最大显存 (MB)</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">CycleNet</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1.77</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.447</td>
      <td style="border: 1px solid #ddd; padding: 8px;">91.37</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">Fredformer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">4.46</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1512.32</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">FilterNet</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1.63</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.456</td>
      <td style="border: 1px solid #ddd; padding: 8px;">79.51</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">iTransformer</td>
      <td style="border: 1px solid #ddd; padding: 8px;">2.44</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.482</td>
      <td style="border: 1px solid #ddd; padding: 8px;">275.12</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 8px;">SOFTS</td>
      <td style="border: 1px solid #ddd; padding: 8px;">2.00</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.466</td>
      <td style="border: 1px solid #ddd; padding: 8px;">183.95</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; background-color: #e6f7ff; font-weight: bold;">
      <td style="border: 1px solid #ddd; padding: 8px;">TimeEmb (本文模型)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">1.61</td>
      <td style="border: 1px solid #ddd; padding: 8px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 8px;">82.36</td>
    </tr>
  </tbody>
</table>



表 8：我们在 ETTh2 数据集上为 Fredformer 和 CycleNet 配备了 TimeEmb。这带来了稳定的性能（MSE）提升，而额外的训练成本微乎其微。


<table style="border-collapse: collapse; width: 80%; border: 1px solid #ddd; text-align: center; margin: 0 auto; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">预测步长</th>
      <th style="border: 1px solid #ddd; padding: 8px;">96</th>
      <th style="border: 1px solid #ddd; padding: 8px;">192</th>
      <th style="border: 1px solid #ddd; padding: 8px;">336</th>
      <th style="border: 1px solid #ddd; padding: 8px;">720</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd; background-color: #fafafa;">
      <td rowspan="6" style="border: 1px solid #ddd; font-weight: bold; padding: 8px; vertical-align: middle;">Fredformer</td>
      <td style="border: 1px solid #ddd; padding: 6px;">Fredformer</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.371</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.415</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">+TimeEmb</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.358</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.360</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.387</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px; font-style: italic;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">1.4%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">3.5%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">5.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">6.7%</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">原参数量</td>
      <td style="border: 1px solid #ddd; padding: 6px;">32,820,131</td>
      <td style="border: 1px solid #ddd; padding: 6px;">33,465,731</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9,894,911</td>
      <td style="border: 1px solid #ddd; padding: 6px;">13,656,959</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">现参数量</td>
      <td style="border: 1px solid #ddd; padding: 6px;">32,830,860</td>
      <td style="border: 1px solid #ddd; padding: 6px;">33,476,460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9,905,640</td>
      <td style="border: 1px solid #ddd; padding: 6px;">13,667,688</td>
    </tr>
    <tr style="border-bottom: 2px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">额外参数量</td>
      <td colspan="4" style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">10,729 (0.03%-0.11%)</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; background-color: #fafafa;">
      <td rowspan="6" style="border: 1px solid #ddd; font-weight: bold; padding: 8px; vertical-align: middle;">CycleNet</td>
      <td style="border: 1px solid #ddd; padding: 6px;">CycleNet</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.285</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.373</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">+TimeEmb</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.277</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.351</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.399</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">0.415</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px; font-style: italic;">提升</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">2.8%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">5.9%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">5.2%</td>
      <td style="border: 1px solid #ddd; padding: 6px; font-weight: bold; color: green;">8.4%</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">原参数量</td>
      <td style="border: 1px solid #ddd; padding: 6px;">99,080</td>
      <td style="border: 1px solid #ddd; padding: 6px;">148,328</td>
      <td style="border: 1px solid #ddd; padding: 6px;">222,200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">419,192</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">现参数量</td>
      <td style="border: 1px solid #ddd; padding: 6px;">109,809</td>
      <td style="border: 1px solid #ddd; padding: 6px;">159,057</td>
      <td style="border: 1px solid #ddd; padding: 6px;">232,929</td>
      <td style="border: 1px solid #ddd; padding: 6px;">429,921</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">额外参数量</td>
      <td colspan="4" style="border: 1px solid #ddd; padding: 6px; font-weight: bold;">10,729 (2.56%-10.83%)</td>
    </tr>
  </tbody>
</table>



![](images/9c445c36fce2103d2d835abe831eb00446629efb10c2e810279b867f1ea8f19d.jpg)  
(a) 不同嵌入长度的结果

![](images/d27fb0ecf83d599e19074276f37cec5f9092be7cbbf5f6300b975fd1898f1993.jpg)  
(b) 不同损失权重的结果  
图 6：超参数分析。

## D.3.2 TimeEmb 中关键组件的消融结果

在本节中，我们通过修改或移除 TimeEmb 中的特定组件，构建了模型的不同版本。结果见表 11。Random：在训练和测试之间对嵌入库进行随机初始化。Zero/Mean：将嵌入库分别固定为全零或全局均值。$" \mathrm { w } / \mathbf { 0 } X _ { s } "$ 表示移除时不变嵌入 $X _ { s }$。$\because \mathbf { w } / \mathbf { o } \varkappa _ { \omega } "$ 表示移除频率滤波器 $\mathcal { H } _ { \omega }$。"w/o RevIN" 是指移除可逆实例归一化。研究表明，时不变分量嵌入对 TimeEmb 的贡献最大。

表 9：消融实验结果。E 包含振幅前 k 位的频率分量。最佳结果加粗表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">k = 5</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">k = 15</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">k = 30</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">k = 40</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">k = 49(full)</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.375</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.374</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.371</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.425</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.419</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.459</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.486</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.473</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.467</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.460</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.438</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.430</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.428</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.167</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.165</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.244</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.238</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.236</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.226</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.330</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.292</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.328</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.326</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.273</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.315</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.314</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.310</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.158</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.202</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.155</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.195</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.152</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.206</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.244</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.205</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.203</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.200</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.341</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.339</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.336</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.272</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.157</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.145</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.140</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.236</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.161</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.255</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.156</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.188</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.185</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.278</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.179</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.273</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.231</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.315</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.229</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.316</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.220</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.307</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.214</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.302</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.187</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.183</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.176</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
    </tr>
  </tbody>
</table>



表 10：消融实验结果。E 包含通过低通滤波（过滤比例为 γ）过滤后的频率分量。最佳结果加粗表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">γ = 0</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">γ = 0.3</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">γ = 0.6</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">γ = 0.9</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">γ = 1</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.370</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.368</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.459</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.471</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.460</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.252</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.227</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.228</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.227</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.226</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.332</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.382</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.274</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.182</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.221</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.191</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.191</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.228</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.202</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.201</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.201</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.200</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.260</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.356</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.343</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.339</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.336</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.238</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.237</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.178</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.141</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.236</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.139</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.138</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.233</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.184</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.158</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.156</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.175</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.316</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.214</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.301</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.211</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.210</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.298</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.201</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.170</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
    </tr>
  </tbody>
</table>



## D.4 超参数分析

我们通过实验评估了 TimeEmb 的核心超参数对预测性能的影响，包括嵌入数量 M 和损失权重 α。根据图 6 中展示的性能表现，针对每种设定分别对这些超参数进行了调整。完整结果可参见表 12 和表 13。

对于嵌入库 E 的嵌入数量，我们设置 $M \in \{ 6 , 1 2 , 2 4 , 9 6 \}$ ，结果如图 6 所示。从结果中可以观察到：（1）M 设置的变化对模型性能的影响微乎其微，突显了 TimeEmb 的鲁棒性及其在最少人工调参下表现优异的能力，这使得它既易于使用又便于部署。（2）不同的数据集由于其不同的周期性模式而表现出截然不同的特征。对于 ETTm2 数据集，当 M = 96 时达到最佳性能，因为更短的时间间隔需要更细粒度的嵌入来捕捉时间模式。对于 ETTh1 和 Weather 数据集，M = 24 取得了最优结果，能有效捕捉数据复杂性并防止过拟合。

表 11：关键模块的消融实验结果。最佳结果加粗表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">模型</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">ETTh1</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">ETTm2</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">天气 (Weather)</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">电力 (Electricity)</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">TimeEmb</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.150</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.226</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.457</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.460</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.336</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">Random</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.407</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.415</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.318</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.443</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.351</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.354</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.499</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.471</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.361</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.314</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.321</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.393</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.550</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.524</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.434</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.370</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.455</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.477</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.338</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.305</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">Zero</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.405</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.414</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.198</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.452</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.442</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.351</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.279</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.352</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.497</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.360</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.314</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.321</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.547</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.523</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.450</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.475</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.462</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.337</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.305</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">Mean</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.397</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.410</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.204</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.175</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.221</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.444</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.439</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.327</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.220</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.261</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.349</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.488</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.328</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.364</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.296</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.537</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.517</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.348</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.374</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.453</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.467</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.350</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.252</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">w/o.Xs</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.376</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.252</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.182</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.221</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.178</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.237</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.228</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.184</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.470</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.295</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.332</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.200</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.487</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.471</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.356</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.347</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.316</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.431</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.274</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.317</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.282</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.201</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.281</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">w/o.Hω</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.391</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.244</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.233</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.421</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.229</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.201</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.155</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.451</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.441</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.286</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.324</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.259</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.492</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.474</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.341</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.337</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.210</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.238</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">w/o. RevIN</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.403</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.147</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.191</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.235</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.450</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.352</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.194</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.252</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.485</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.449</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.248</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.271</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.517</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.511</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.516</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.482</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.326</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.346</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.213</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.306</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.459</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.457</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.355</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.392</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.229</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
    </tr>
  </tbody>
</table>



对于损失权重，我们使用 $\alpha \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ 进行了实验。图 6 的结果表明，合适的 α 值可以提升 TimeEmb 的性能。结合时域和频域损失的效果要优于仅使用时域损失 $( \alpha = 0 )$ ，这表明整合来自这两个领域的特征信息可以增强 TimeEmb 捕获时间序列数据中多样化模式的能力。

我们考察了几个关键超参数对 TimeEmb 性能的影响：嵌入库中的嵌入数量（记为“M”）和优化目标中的损失权重（记为“α”）。详细结果在表 12 和表 13 中给出。这表明适当的超参数设置可以增强 TimeEmb 的性能。

## D.5 所学嵌入的可视化

我们在图 7 中展示了时不变分量嵌入。术语“hour x”中的“x”代表输入序列最后一个时间步的小时索引（hour-index）。图 7 描绘了从不同数据集和通道中学习到的独特嵌入。例如，图 7 (a) 展示了在电力（electricity）数据集通道 302 中学习到的时不变分量，其中小时索引为 5。相比之下，图 7 (b) 显示了 Weather 数据集通道 15 中与输入序列对应的时不变分量，其小时索引为 2。这些源自全局序列的嵌入捕捉了时不变分量，为模型提供了至关重要的补充信息，从而有助于更好地理解时间序列数据中的稳定模式。

表 12：时不变嵌入数量 M 的影响。最佳结果加粗表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">M = 6</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">M = 12</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">M = 24</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">M = 96</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.372</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.370</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.371</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.390</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.419</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.422</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.420</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.458</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.481</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.462</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.460</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.461</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.435</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.428</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.425</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.427</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.426</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.167</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.246</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.164</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.163</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.242</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.232</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.231</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.289</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.227</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.226</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.291</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.328</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.290</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.327</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.323</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.387</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.385</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.383</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.381</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.270</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.313</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.159</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.199</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.153</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.157</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.198</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.205</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.202</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.201</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.246</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.284</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.338</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.339</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.336</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.342</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.339</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.349</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.242</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.269</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.140</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.235</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.138</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.157</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.249</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.155</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.248</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.173</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.299</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.212</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.300</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.297</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.296</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.262</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
    </tr>
  </tbody>
</table>



表 13：损失权重 α 的影响。最佳结果加粗表示。


<table style="border-collapse: collapse; width: 100%; border: 1px solid #ddd; text-align: center; font-size: 0.9em;">
  <thead>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">数据集</th>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 8px;">H</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">α = 0</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">α = 0.25</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">α = 0.5</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">α = 0.75</th>
      <th colspan="2" style="border: 1px solid #ddd; padding: 8px;">α = 1</th>
    </tr>
    <tr style="background-color: #f2f2f2; border-bottom: 2px solid #ddd;">
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MSE</th>
      <th style="border: 1px solid #ddd; padding: 6px;">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTh1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.382</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.402</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.389</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.388</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.366</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.367</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.387</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.423</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.424</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.419</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.418</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.417</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.417</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.416</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.444</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.463</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.440</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.438</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.437</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.436</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.459</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.460</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.457</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.468</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.462</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.472</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.464</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.474</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.465</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.432</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.433</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.428</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.426</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.428</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.426</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.428</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.426</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.429</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.426</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">ETTm2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.170</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.251</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.166</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.245</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.165</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.244</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.164</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.243</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.164</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.243</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.293</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.230</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.228</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.227</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.227</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.294</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.333</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.325</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.286</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.285</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.323</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.287</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.324</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.405</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.398</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.386</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.384</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.382</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.383</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.382</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.276</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.319</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.311</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.266</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.310</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.308</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.265</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.309</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">天气 (Weather)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.197</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.151</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.193</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.150</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.190</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.203</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.243</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.202</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.240</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.201</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.201</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.201</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.288</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.260</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.285</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.260</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.283</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.259</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.282</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.344</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.342</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.342</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.340</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.342</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.339</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.342</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.339</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.241</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.268</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.265</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.239</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.263</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.238</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.262</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td rowspan="5" style="border: 1px solid #ddd; font-weight: bold; padding: 8px;">电力 (Electricity)</td>
      <td style="border: 1px solid #ddd; padding: 6px;">96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.234</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.136</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.231</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.137</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.232</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">192</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.155</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.250</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.153</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.246</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.154</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.247</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">336</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.172</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.267</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.170</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.171</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.264</strong></td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">720</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.211</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.303</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.208</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.209</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.297</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.210</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.298</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.210</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.298</td>
    </tr>
    <tr style="border-bottom: 1px solid #ddd;">
      <td style="border: 1px solid #ddd; padding: 6px;">avg</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.169</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.264</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.167</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.168</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><strong>0.260</strong></td>
    </tr>
  </tbody>
</table>



![](images/cab7cc1712c577ce8f79c2187a3ffaa3756abec77ca6d5f860fc1a2da8e6800b.jpg)  
(a) ECL

![](images/7da82a151f0ee271b63efe8fa6ecbdf045df35b851aa9c31ca6e166dd681d0c5.jpg)  
(b) Weather

![](images/258bd082e6eee55d361d796bb9c1b507a1bed52f638bd2960675949e8a3450af.jpg)  
(c) Traffic

![](images/e8e25d50eee66dcfcc523e5a1b62454625fd6af574a91a800dabfe0071516c74.jpg)  
(d) ETTh1

![](images/9764bc4523f22358db36ee9dded4875b06d94e65b2f2b15f68401ed18dc27682.jpg)  
(e) ETTm1

![](images/d822c9fb0c719cda061636db9e23b56a622cc6777a0ba29833d4623495330444.jpg)  
(f) ETTm2  
图 7：所学习的时不变嵌入 $X _ { s }$ 的可视化

## D.6 预测可视化

我们在 ETTh2、Electricity 和 Traffic 数据集上展示了预测案例，如图 8 所示。预测结果与真实值（ground truth）高度一致，证明了 TimeEmb 能够捕捉这些数据集复杂的时序依赖关系。

![](images/8a2a92fa221c681dc454470d624fdb0dbd1938aa9b519832c0f73e9fde9b4f56.jpg)  
(a) Electricity

![](images/67e0c6d92b328c5b1526fbd29fd9919be911c4c3b9192f0387bf857ed6eb7bae.jpg)  
(b) ETTh2

![](images/e9ee31fff39143bfff592b71fe22b552fad26c4c4641fba3630d1aaeefd6280f.jpg)  
(c) Traffic  
图 8：TimeEmb 预测与相应真实值的可视化。

## E 局限性

尽管 TimeEmb 在频域时间序列建模中表现出了强大的性能和可解释性，但仍存在若干局限性。首先，目前的设计采用了固定分辨率的嵌入库，这限制了其自适应捕获跨多个时间粒度（例如，小时、天、周）稳定模式的能力。一种更灵活的多尺度时不变表示机制将进一步增强其学习多周期结构的能力。其次，目前的嵌入结构是离散的，这可能会限制其建模持续演变周期性的能力。将嵌入公式扩展到连续或核函数化（kernelized）表示，可以实现在未见时间插槽（unseen temporal slots）上更平滑的泛化。