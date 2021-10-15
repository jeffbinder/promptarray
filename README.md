# Boolean Prompting for Neural Text Generators

Neural text generators like the [GPT](https://github.com/openai/gpt-2) models promise a general-purpose means of manipulating texts. These models are trained to perform a simple task: given the beginning of a text, the model tries to predict what word will come next. By applying this procedure repeatedly, a model can generate a _completion_, meaning a continuation of the text that starts with a given prompt. Applying this technique to particular problems requires constructing a prompt that induces the model to produce the desired output. This development represents a step toward a dream the computer scientist Jean E. Sammet expressed over half a century ago: [programming a computer in English](https://dl.acm.org/doi/abs/10.1145/365230.365274).

Designing reliable prompts, however, is a complex matter. The emergence of text generators has led to the practice of [prompt engineering](https://arxiv.org/abs/2107.13586)—that is, techniques (some automated, some manual) for designing better language model inputs. Some researchers have also developed new approaches to text generation that depart from the basic prompt-in, completion-out paradigm. One [proposed approach](https://arxiv.org/pdf/2103.00453.pdf) uses a secondary prompt to indicate forms of "bias" that are to be discouraged in the output. A related technique is [calibration](https://arxiv.org/pdf/2102.09690.pdf), which adjusts the probability distribution based on the output given a generic input.

This repo implements the rudiments of what I am hoping will become a broader set of techniques for controlling text generators. Rather than entering a single text prompt that is fed directly into the model, one enters an expression that can incorporate the Boolean operators `&`, `|`, and `~`, meaning *and*, *or*, and *not*, which guide how the neural network is run. In essence, this creates hybrid of a text generator and a programming language, making it easy to compose arrays of multiple prompt variants and experiment with new ways of manipulating the numerical outputs of language models.

As an illustration, consider the following prompt, which one might use as a way of generating descriptions of rabbits:

> Scientists recently discovered a new species of rabbit. Here is a description of it:

This prompt works well enough, but it is limited in its ability to exploit the information present in the model. One issue is that there are multiple words for this species: it could also be called a "bunny." Feeding in "bunny" instead of "rabbit" would produce slightly different results, because different patterns exist in the use of these words in the training data. What if we don't care about these differences? Using the *or* operator, one can construct a single prompt that considers both options:

> Scientists recently discovered a new species of {rabbit|bunny}. Here is a description of it:

The `|` indicates *or*; the brackets delimit the text that is affected by the operator and are otherwise ignored. This prompt causes the program to choose words that may be predicted based on either the "rabbit" or the "bunny" variant of the prompt, effectively ignoring the difference between the two words.

The program also includes an *and* operator, which selects predictions that are common to the different variants:

> Scientists recently discovered a new species of {rabbit&fish}. Here is a description of it:

This prompt tends to produce output that is not especially specific to any animal, or at least that applies to both rabbits and fish.

Perhaps the most powerful part of this system is the *not* operator (`~`), which can be used to discourage the model from producing the sort of output induced by certain input. Note that `~` is a binary operator; `A~B` may be read as "`A` but not `B`." For example, one might want to generate a description of a fish while avoiding on particularly popular type of fish:

> Scientists recently discovered a new species of {fish~shark}. Here is a description of it:

This generates predictions based on the "fish" variant while discouraging ones that are predicted based on the "shark" variant.

The negation operator has a number of potential uses. One technique is to place an empty string on either side of the operator; this can be used to boost or diminish the effect of a particular piece of text on the model. For instance:

> Scientists recently discovered a new species of rabbit{~ in the United States}. Here is a description of it:

This will bias the model against saying that the rabbit was found in the United States and toward saying it was found in other places. By putting it on the other side, one can amplify the phrase's effect:

> Scientists recently discovered a new species of rabbit{ in the United States~}. Here is a description of it:

This has the opposite effect compared to the above, placing a heightened emphasis on the US context.

If you want to include any of the special characters in the prompt, you can escape them with `\`. The system allows operations to be combined arbitrarily, producing complex expressions like the following:

> {~Content warning: violence. }The following is a {true|faithful} \\& vivid account of what happened to me when I was abducted by {aliens&government agents&{foreign|Russian} spies}.

Below I present some preliminary findings suggesting that this system can improve the performance of language models at certain tasks—by one benchmark, LAMBADA, even bringing relatively small models like the 774B parameter GPT-2 close to the reported performance of the massive GPT-3. It also gives the user new ways of controlling the output of text generators, opening the potential for more systematic, nuanced, and creative approaches to what is, at present, often a matter of trial and error.

## What It Does (and Why, and How)

This system was inspired in part by my research for my forthcoming book on the history of algorithms and, in particular, from a reconsideration of [George Boole](https://georgeboole.com/boole/)'s work on algebraic logic. The logic system that bears Boole's name (at least as it appears in programming languages like Python) is an arithmetic of two values, `true` and `false` or 1 and 0. This is not, however, how Boole's original system worked; his variables could have non-numeric values that represented classes of things such as "trees" and "sheep," and he used the *and* and *or* operators algebraically, not as operations to be executed. Like Boole's work, this system implements *and,* *or,* and *not* in a way that incorporates far more semantic content than the two-valued data type of standard programming languages. Instead of applying these operators to truth values, it applies them to probability distributions that represent (however inadequately) something of the complexity of language.

This project also takes inspiration from some old insights from linguistic anthropology. Structuralists such as [Claude Lévi-Strauss](https://press.uchicago.edu/ucp/books/book/chicago/R/bo3614777.html) maintained that language is based on difference: in order to understand what it means for something to be raw, we also must understand the meaning of "cooked." Yet it is not always self-evident which words are opposites. Perhaps it is clear that the opposite of day is night, but what is the opposite of a narrative? Is it, as [Lev Manovich once posited](http://mfj-online.org/journalPages/MFJ34/Manovich_Database_FrameSet.html), a database? A scientific analysis? A photograph? Silence? The *not* operator makes it possible to specify which opposite one has in mind, whereas *or* does the opposite, indicating that a particular distinction is not important.

My implementation of these operations is somewhat like a programming language, in that expressions are parsed into a syntax tree which is then compiled into a sequence of executable operations. The semantics, however, are very different compared to standard programming languages. Continuing my [longstanding interest](http://jeffreymbinder.net/208/homespring) in unusual programming languages, this project creates something of a hybrid between Boolean expressions and the English language. In executing programs, the system takes (and I nod here to [G. W. Leibniz](https://global.oup.com/academic/product/leibniz-dissertation-on-combinatorial-art-9780198837954?cc=us&lang=en&)) a combinatorial approach, generating all possible variants of the prompt based on the alternatives set out by the operators. It then runs the model on all of these variants and aggregates the resulting probability distributions based on the indicated operations.

For example, consider the following, very simple example:

> Hello|Goodbye

This expression provides two options: the text is either "Hello" or "Goodbye." In order to interpret it, the software first runs the model on each of these prompts. The result is two probability distributions over possible next tokens, which constitute the model's predictions for each prompt. These distributions are combined by adding them together (an implementation of the *or* logic), then normalized to produce a new distribution that is used in generation. As the program generates more words, it continues to consider both prompts so that the Boolean operator affects all of the output.

Now consider a more complex expression:

> {Hello|Greetings}. {We&{I~They}} welcome you to|Welcome to

Since there are alternatives at multiple points in the prompt, all possible combinations must be considered. The full expansion, in this case, includes six variants:

> 0: Hello. We welcome you to  
> 1: Greetings. We welcome you to  
> 2: Hello. I welcome you to  
> 3: Greetings. I welcome you to  
> 4: Hello. They welcome you to  
> 5: Greetings. They welcome you to  
> 6: Welcome to

Note that the text to which the *not* operator applies—the word "They" as an opening for the second sentence—must still be considered as an option so that its effects can be discouraged. In order to combine the outputs, the software generates a simple program that indicates the order in which the operations must be performed:

> 0: 0 |= 1  
> 1: 2 |= 3  
> 2: 4 |= 5  
> 3: 2 ~= 4  
> 4: 0 &= 2  
> 5: 0 |= 6

This means that the values for prompts 0 and one are combined with the `|` operation and the result stored at position 0; the same is then done for 2 and 3, and so forth. In the end, the final result always ends up at position 0.

There is some room for debate as to how, mathematically, the probabilities should actually be calculated, and I am still working on developing the best methods. At present, the program uses simple addition for *or* and multiplication for *and*. This corresponds to Boole's original correspondence between logic and probability. The *not* operator, however, is somewhat different. An obvious interpretation would make this operation correspond to subtraction; this is how Boole's system works. But subtraction does not work in this case, because subtracting the probabilities destroys the information needed to produce coherent text. The same problem applies to division. After experimenting with a number of options, I found the best results using this formula:

> `p_{A~B} = p_A / sqrt(p_B)`

The square root reduces the relative effect of B, discouraging words that it encourages while still retaining something of the overall probability structure produced by A.

This technique may be used with expressions of arbitrary complexity. All of the variants are fed into the model in one batch, so a more complex prompt expression does not take much more time to run than a simple one, at least on a GPU. However, expressions will use up more GPU memory the more complex they are.

The main logic of this system, including the compiler and code for executing programs, appears in `program.py`. The file `generation_utils.py` amends the corresponding file from the Transformers package to invoke this system. The easiest way to get started is to run `generate.py`, whose source can be edited to change the settings. The system works in basically the same way as the regular Transformers generator, but instead of passing a sequence of token ids to `generate()`, you pass in a string containing the Boolean expression. To run this code, you will need to have recent Git master versions of PyTorch and Transformers installed.

The code in this repo implements Boolean prompting for autoregressive text generators; it can be used with GPT, GPT2, XLNet, XLM, CTRL, and Transformer XL models, either with or without finetuning. The broader technique I describe is not, however, specific to this type of generator. In my project [A Hundred Visions and Revisions](https://github.com/jeffbinder/visions-and-revisions), first published in March 2020, I incorporated a technique similar to the *not* operator (there called "strong topic bias") into a non-autoregressive text rewriting procedure based on [BERT](https://github.com/google-research/bert)-style masked language models.

## Experiments

I am continuing to develop and experiment with this method, but I do have some preliminary results.

### Discouraging Words

An obvious application of the *not* operator would be to prevent (or at least discourage) the model from doing certain things. Evaluating success at this goal is difficult, since judgments about the quality and meaning of texts can be debatable. However, it is possible to test the effect in a rough way by checking how often certain, generally undesirable words appear.

Consider the following prompt:

> Prompt A. Scientists recently discovered a new species of snake. Here is a description of it:

This prompt does a fairly good job of producing text that resembles scientific (or pseudo-scientific) descriptions of snake species. Quite often, however, the output contains statements that belie the nature of snakes, such as saying the animal has fur or legs. Such problems could in theory be rectified with a better model. However, it is also possible to make better use of the information that is present in the current model.

One might think to do this simply by altering the wording of the prompt so as to incorporate the information that, for instance, snakes lack legs:

> Prompt B. Scientists recently discovered a new species of snake, an animal without legs. Here is a description of it:

As an experiment, I ran GPT-2 XL a thousand times with each of these prompts and counted how many of the outputs contained one of three words associated with legs. Here are the results, including p-values computed using Barnard's exact test. As the results show, this method does not have the expected effect.

| Words | Prompt A | Prompt B | p A\<B |
| --- | --- | --- | --- |
| leg/legs/legged | 101/1000 | 646/1000 | ~9e-149 |

(Note that these results count the number of outputs that contain at least one of the words, ignoring how many times the words occur. This is because the individual words in the output are not statistically independent: after the generator has mentioned legs once, it is probably more likely to do so again.)

With the added phrase "an animal without legs," the generator actually mentions legs over six times more often. Some of these instances come from the model repeating the statement that the snake does not have legs, a nuance that is lost in this simple word counting method. But this is not always the case, as one can see by examining the actual results (`discouraging-results/snake no legs-v2`). I reviewed the first 200 outputs and found that, of the 136 that contain the word "legs," around 54 suggest that the animal does indeed have legs (although some of them also, contradictorily, say that it has none). One of the outputs, for instance, begins this way:

> This new snake is the only snake with legs. This is a male, which is the largest snake in its genus (Eunectes). The snake is about 2.5 meters (8 feet) long.The snake is named "Eunector," which is Latin for "snake with legs."

This is an instance of a [much-discussed](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00298/43535/What-BERT-Is-Not-Lessons-from-a-New-Suite-of) problem language models have in dealing with negation. The problem stems in part from the nature of the predictive task: ordinarily, one wouldn't mention that the animal lacked legs unless legs had some relevance. However, it also stems from a general limitation of GPT-2's ability to understand negative words such as "without."

Boolean prompting provides an alternative way of specifying negation that produces much better results:

> Prompt C. Scientists recently discovered a new species of snake{~ with legs}. Here is a description of it:

The results are as follows:

| Words | Prompt A | Prompt C | p A\>C |
| --- | --- | --- | --- |
| leg/legs/legged | 115/1000 | 48/1000 | ~2e-8 |

As the results show, the method does not work perfectly—it only reduces the references by about half. But the effect is clear, indicating that the operator is altering the behavior of the generator in the desired direction.

This technique is not the only way the *not* operator may be applied. Another approach is to utilize both sides of the operator, indicating both what the animal is and what it is not. This technique can be used to discourage those irksome references to hair and fur:

> Prompt D. Scientists recently discovered a new species of {snake~mammal}. Here is a description of it:

These are the results:

| Words | Prompt A | Prompt D | p A\>D |
| --- | --- | --- | --- |
| fur/furred/furry | 19/1000 | 4/1000 | ~0.0008 |
| hair/hairs/haired/hairy | 61/1000 | 36/1000 | ~0.005 |

As these results show, adding the Boolean operator to the prompt led to a significant decrease in words referring to the mammalian traits of fur and (to a somewhat lesser extent) hair.

An obvious application of this technique would be in discouraging the model from generating text with offensive, violent, or otherwise undesirable qualities. As such, this method would be conceptually similar to the ["self-debiasing"](https://arxiv.org/pdf/2103.00453.pdf) method proposed by Schick, Udupa, and Schütze. I would caution, however, that the technique's ability to discourage offensive text is only as good as the model's ability to distinguish offensive from non-offensive, which is to say (at least in the case of GPT-2) not that great. Such efforts also run into political difficulties: people do not all agree on what is offensive, and it is a complex matter to determine exactly what stance on this issue has wound up embedded in the model. Thus, while putting something like "{~Content warning: racism. }" into a prompt might mitigate the problem to some extent, it should not be taken as a solution.

A more immediate application is improving the ability of text generators to perform certain predictive tasks, as I discuss in the next section.

### LAMBADA

The [LAMBADA](https://zenodo.org/record/2630551#.YWb8Iy-cbOQ) benchmark tests the ability of models to account for long-range dependencies in texts. LAMBADA is based on a collection of excerpts from novels, selected such that humans are able to guess the final word, but only when they are given the whole passage. The model is supplied with the passage absent the final word, and must predict what that word is.

The LAMBADA benchmark is usally scored in two ways, the accuracy of predictions (measured as a percentage) and perplexity, which measures how well the probabilities produced by the model align with a text. Since the present technique alters the way the continuation is generated without altering the model itself, perplexity is not clearly applicable, so I have focused solely on accuracy.

The creators of GPT-2 [report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that the largest version of their model attained an accuracy of **63.24**. Accoring to [this discussion](https://github.com/openai/gpt-2/issues/131), this experiment was based on predicting the final token of the text, not the final word. Since GPT-2 sometimes splits words into multiple tokens, this is an easier task than predicting whole words. I evaluated my method with both versions of the task to enable fair comparisons with other results.

The performance of GPT-3 was tested using several different approaches. The authors advocate a "few-shot" approach, in which the model is primed with several completed examples before being presented with a problem; the [reported accuracy](https://arxiv.org/abs/2005.14165v4) with this approach is **86.4**. This approach is not, however, generally workable with smaller models, so I did not attempt to replicate it here. GPT-3's reported accuracy with the zero-shot approach, which I have employed in this experiment, is **76.2**.

I tried several formulae for constructing the prompt. Here "context" refers to the passage, excluding the final word or token that is to be predicted.

1. \<context\>
2. \<context\>~  
3. \<context\>~[...]\<last word of context\>  
4. \<context\>~[...]\<last phrase of context\>  
5. \<context\>~[...]\<last sentence of context\>  
6. \<context\>~[...]{\<last sentence\>|\<last word\>}  
7. \<context\>~[...]{\<last sentence\>|\<last phrase\>}

The rationale is to discourage the model from making predictions based solely on the later parts of the prompt while ignoring the earlier ones. Phrases are delineated using the following regex: `[,.:;?!"“”]`. Sentence boundaries are determined using the Punkt sentence tokenizer.

These are the results for token prediction:

| Model | # params | Baseline | Blank | Last word | Last phrase | Last sentence | Last sentence or word | Last sentence or phrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 117M | 48.03 | 58.63 | 62.37 | 65.61 | 67.71 | 67.81 | 67.81 |
| gpt2-medium | 345M | 56.76 | 61.50 | 70.66 | 72.39 | 74.48 | 74.29 | 74.50 |
| gpt2-large | 774M | 60.95 | 67.86 | 74.13 | 75.65 | 77.45 | 77.51 | 77.37 |
| gpt2-xl | 1558M | 63.98 | 70.17 | 76.38 | 77.47 | 78.83 | 79.22 | 79.02 |

These are the results for whole-word prediction:

| Model | # params | Baseline | Blank | Last word | Last phrase | Last sentence | Last sentence or word | Last sentence or phrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 117M | 34.10 | 45.33 | 50.73 | 54.98 | 58.16 | 58.33 | 58.24 |
| gpt2-medium | 345M | 44.79 | 47.72 | 62.10 | 63.46 | 66.93 | 66.97 | 67.07 |
| gpt2-large | 774M | 50.05 | 56.61 | 66.02 | 67.79 | 70.97 | 70.87 | 70.72 |
| gpt2-xl | 1558M | 53.87 | 59.13 | 68.60 | 69.90 | 72.39 | 72.85 | 72.54 |

You can replicate these results using the `lambada_score.py` script.

To some extent, these techniques work by exploiting the specific nature of the test dataset. LAMBADA was designed for testing a model's ability to find relevant information that occurs in previous sentences. By boosting the effects of the earlier parts of the prompt, the negation operator ensures that the model considers the prompt as a whole.

From an AI perspective, this approach might be seen as a way of gaming the system. Rather than improving the model, the program is sifting through the output to find the predictions that best suit the nature of the LAMBADA test. Yet this sifting is only a problem if we insist that the machine be capable of both determining the nature of the task and performing it without human intervention. If we see language models not as [a foundation for general intelligence](https://arxiv.org/pdf/2108.07258.pdf), but rather as a practical means of performing computations, then designing prompt expressions that suit the task at hand is a useful technique. Boolean prompting is an alternative to AI purism, an approach that enables human and machine to work together.

There is also reason to think that at least some of the improvement stems from something other than the specific nature of the task. In all but one case, simply adding a `~` operator to the end of the input improved the performance by more than five percentage points. This intervention makes no assumptions about the task at hand; it merely encodes the prior that the prompt contains some information that is relevant to the generative task. As an explanation of the increased performance, I would hypothesize that the model is conflating the overall frequency of a word with its probability in a particular context; the `~` operator induces the system to focus more on the context. This technique could potentially be applied to any number of tasks.
