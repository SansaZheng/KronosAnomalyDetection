# Kronos 原生输入核对

现有预训练tokenizer的`d_in=6`，其六维含义为open、high、low、close、volume、amount。

此前20组基线结果不是Kronos结果；原生Kronos实际运行结果保存在相邻`kronos/`目录。

本次逐项核对发现10个基线数据版本均没有真实OHLCVA通道。即使BankSim恰好有6个模型输入特征，也不等于6维金融K线。不能仅凭维数匹配就送入原生模型。逐项状态见`coverage.csv`。

如果使用逐特征代理序列，需要明确：这会将非市场特征编码到预训练K线模型的输入槽中，属于新的、域外的适配实验；不能声称它是原始Kronos金融K线方法。用户已明确允许该适配；相应抽样实验单独保存在 `../kronos_proxy/`，与原生Kronos分开报告。已有基线和原生Kronos结果保持原样。
