# Relative action-quality hypothesis

Frozen before inspecting the absolute critic72 result. User approved a few further sequential experiments. This is candidate2 after data-diversity candidate1, not a seed/architecture sweep.

Question: does directly learning the difference in final quality between actions improve the decision-relevant signal, avoiding the nuisance of predicting absolute question difficulty?

Reuse exactly the same72fit/12monitor questions and error exclusions as critic72. These monitor questions become development evidence across the two prespecified candidates; not untouched independent confirmation. Exclude old12monitor. No new collection. Same FutureCritic architecture/initialization seed20261077,100AdamW updates lr.001,weight_decay.01,gradientclip1,finalcheckpoint only. Frozen reader and zero actor remain unchanged.

Replace absolute quality MSE with mean squared error of Q_F(a)-Q_F(b) versus observed terminalF1(a)-terminalF1(b), over all distinct observed legal action pairs at each state. Equal question weights, equal states within question, equal pairs within state. Cost loss stays generated-token MSE divided by100 squared. Do not introduce learned absolute-quality calibration or deploy Q argmax: common quality offset is unconstrained by this loss. No future information enters prefix features.

Compare on same12 to frozen absolutecritic72 and fit72 per-action means. Require >=10 unequalF1 pairs over>=5questions, action-gap MSE strictly below BOTH comparators, costMAE strictly below BOTH comparators, signaccuracy>.5. Failure means this candidate cannot proceed to actor guidance. No rescue by changing loss weights,epochs,monitor or gate after results. Absolutecritic72 is fixed reference regardless of its own gate outcome.

Use scripts/probe_decision_relative_critic72.py with reused audited critic72 runner. In its result schema `old` denotes absolutecritic72, not initial36critic; record reference path/hash in manifest/report. Freeze completed data and reference hashes after absolutecritic72 audit and before fitting. No automatic actor launch. Goal still requires actual policy F1+5pp/generated<=70% basicGRPO and useful state-conditioned decisions, then protected independent confirmation.
