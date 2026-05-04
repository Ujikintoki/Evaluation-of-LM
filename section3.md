3 Biases in Language Models (40%)
In this part of the project, you’ll evaluate biases, stereotypes, and associations in Masked LMs concerning social groups using the ”Minimal Pairs” method. This approach focuses on various social domains
such as race, gender, and political ideology. You’ll experiment with two masked language models listed
in Section 1 or other models you prefer.
To start with, you will choose a group that is important to you (e.g., Hispanic, female, Chinese,
LGBT+) and study the social domain containing that group throughout this part. You will measure
biased stereotypes related to that domain, especially when mentioned in the text. For example, if the
female group is important to you, you can select gender/gender identity as the domain you study in
this part [4].
Methodology –Minimal Pairs: Previous work [3, 4] introduces evaluation methods known as
minimal pairs. In this approach, a model is presented with pairs of sentences that differ by only a
few words, contrasting stereotypical and astereotypical associations. The logic is that discrepancies
in the probabilities assigned by a language model to these sentences reveal whether the model has a
bias or preference for a particular stereotype. Ideally, an unbiased language model should assign equal
probabilities to both the stereotypical and astereotypical sentences in each pair.
Dataset and Social Domain Please use the dataset “CrowS-Pairs” curated by [4], which recognizes
nine social domains for detecting bias and stereotypes in language models: race/color, gender/gender
identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status/occupation.
You need to choose a domain that contains the group important to you. You should identify
sentence pairs that encode stereotypes related to this domain to evaluate LMs (and abandon sentence
pairs related to other domains).
Some domains contain much more examples than others, i.e., race/color, gender/gender identity,
socioeconomic status/occupation, nationality, and religion. If you choose one of those, you are allowed
sample 80 examples to conduct the experiments.
Metric Please use the pseudo-log-likelihood proposed in Section 3 in the CrowS-Pairs paper [4].
This metric use Masked LMs to compute pseudo-log-likelihood scores for each sentence. Then, this
metric measures the percentage of pairs for which a model assigns a higher pseudo-log-likelihood to
the stereotyping sentence. A masked LM without any stereotypes shuold achieve the ideal score of
50%.
3.1 Deliverables
Your report should include:
Setup.
1. Social group. The social group and domain you choose. Please also include the total number
of examples you used.
2. Metric. A clear explanation of how the pseudo-log-likelihood metric works.
3. Implementation details: Please also include some important implementation details, such as
model size and hyper-parameters.
4. Experimental details. Include other details for how you set up the experiments.
4
Results.
1. Quantitative Results. Quantitative results for evaluating masked language models on your
bias evaluation data. You should compare the biases of the models you choose.
2. Case Study. Three examples of sentence pairs should be depicted in a figure or table, demonstrating specific biases in both models.