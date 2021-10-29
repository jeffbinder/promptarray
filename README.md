# PromptArray: A Prompting Language for Neural Text Generators

Neural text generators like the [GPT](https://github.com/openai/gpt-2) models promise a general-purpose means of manipulating texts. These models are trained to perform a simple task: given the beginning of a text, the model tries to predict what word will come next. By applying this procedure repeatedly, a model can generate a _completion_, meaning a continuation of the text that starts with a given prompt. Applying this technique to particular problems requires constructing a prompt that induces the model to produce the desired output. This development represents a step toward a dream the computer scientist Jean E. Sammet expressed over half a century ago: [programming a computer in English](https://dl.acm.org/doi/abs/10.1145/365230.365274).

Designing reliable prompts, however, is a complex matter. The emergence of text generators has led to the practice of [prompt engineering](https://arxiv.org/abs/2107.13586)—that is, techniques (some automated, some manual) for designing better language model inputs. Some researchers have also developed new approaches to text generation that depart from the basic prompt-in, completion-out paradigm. One [proposed approach](https://arxiv.org/pdf/2103.00453.pdf) uses a secondary prompt to indicate forms of "bias" that are to be discouraged in the output. A related technique is [calibration](https://arxiv.org/pdf/2102.09690.pdf), which adjusts the probability distribution based on the output given a generic input.

This repo implements the rudiments of what I am hoping will become a broader set of techniques for controlling text generators. as opposed to entering a single text prompt that is fed directly into the model, one enters an expression that can incorporate the following operators:

| Operator | Meaning |
| --- | --- |
| A&B | A and B |
| A\|B | A or B |
| A^B | A and not B |
| A/B | A more than B |
| A~B | A as opposed to B |

In essence, this creates hybrid of a text generator and a programming language, making it easy to compose arrays of multiple prompt variants and experiment with new ways of manipulating the numerical outputs of language models. The primary downside is an increase in the use of GPU memory.

Apart from introducing a new syntax, this project suggests a new interpretation Boolean logic. Boolean operators are usually understood in terms of truth values: "A and B" means that A and B are both true. But Boolean logic can also be interpreted in terms of meaning: "big and red" means the category of things that are both big and red. This semantic interpretation, as I hope to show, can be formalized in a way that is actually computable using language models.

As an illustration, consider the following prompt, which one might use as a way of generating descriptions of rabbits:

> Scientists recently discovered a new species of rabbit. Here is a description of it:

Here is some example output:

> The new species, named the New World Rabbit (Oryctolagus cuniculus), is the first to be found in North America since 1872. It is a medium-sized, short haired rabbit, with an average weight of 1.3 kilograms and a height of 1.2 meters (4 feet). The rabbit has dark gray fur, and its body color ranges from white, to black, and sometimes gray. Its ears are large and rounded.

This prompt works well enough, but it is limited in its ability to exploit the information present in the model. One issue is that there are multiple words for this species: it could also be called a "bunny." Feeding in "bunny" instead of "rabbit" would produce slightly different results, because different patterns exist in the use of these words in the training data. What if we don't care about these differences? Using the *or* operator, one can construct a single prompt that considers both options:

> Scientists recently discovered a new species of {rabbit|bunny}. Here is a description of it:

The | indicates *or*; the brackets delimit the text that is affected by the operator and are otherwise ignored. This prompt causes the program to choose words that may be predicted based on either the "rabbit" or the "bunny" variant of the prompt, effectively ignoring the difference between the two words:

> This is a new species of bunnies that lives in an area called "Bunny Hill". This is a very small area, only 1 km2. There is a lot of vegetation and a few trees. The bunny is a small rabbit with white fur, which can grow to about 2 cm in length. The fur on the back is white, while the ears and tail are brownish. The eyes and nose are yellowish, with black spots.

This technique can also be used to create combinations of categories that are very different.

> Scientists recently discovered a new species of {snake|bird}. Here is a description of it:

The result tends to be descriptions of animals that combine the qualities of birds and snakes:

> The new snake is called a Pteranodon, and the scientists believe it was the most primitive snake ever found. It is about 2 meters (7.6 feet) long, has no scales on its head and body, and its tail is only about 3 centimeters (1.5 inches) long, according to a report by National Geographic. It also has a large mouth with teeth like those found in modern snakes, and a large, curved beak.

The program also includes an *and* operator, which selects predictions that are common to the different variants:

> Scientists recently discovered a new species of {snake&bird}. Here is a description of it:

This prompt tends to produce output that is not especially specific to any animal, or at least that can apply to both snakes and birds. If often seems to talk its way around exactly what sort of creature is being described:

> The new species is a new genus and species, named Pteropus vampyrus. It was discovered in the forests of the Sierra Nevada Mountains in California. The new genus and species name honors Dr. Robert Vamp, a noted ornithologist and curator of the Museum of Vertebrate Zoology at the University of Chicago. The new genus name honors Dr Robert V. Pteropus, who discovered this species in the Sierra Nevada mountains of California in the early 1960s.

Perhaps the most powerful operator is the *as opposed to* operator (~), which can be used to discourage outputs that are produced by a certain input. (This is related to the Boolean *not*, although there is a technical difference that I discuss below.) For example, one might want to generate a description of a "serpent" while emphasizing that this is not merely a synonym for "snake":

> Scientists recently discovered a new species of {serpent~snake}. Here is a description of it:

This generates words about a "serpent" while discouraging words that could also apply to a "snake." The result is a description of a different type of serpentine creature:

> The Serpent of the Sea (Sphyrna leucophylla) is a serpent that lives on the ocean floor, and has a long, thin body with a long neck, and a small head with a small mouth. It can be distinguished from other serpent species by its long neck, the fact that it has two pairs of fins, the fact it does not swim, and by the presence in its mouth of a pair of large spines, which are used for grasping and killing prey.

The *as opposed to* operator has a number of potential uses. One technique is to place an empty string on either side of the operator; this can be used to boost or diminish the effect of a particular piece of text on the model. For instance:

> Scientists recently discovered a new species of bison{~ in the United States}. Here is a description of it:

An example:

> This species was discovered in Mongolia by the Mongolian Bison Conservation Project, which has a team in China and in Mongolia to monitor bisons and protect them.The new species was described by a team led by Dr. Zhan-Jin Li of Tsinghua University in Beijing and his colleagues, who have been studying bison for decades. The species was first described in 2003 by a group from Tsinghua.

This will suppress words that place the bison in the United States. By putting it on the other side, one can amplify the phrase's effect:

> Scientists recently discovered a new species of rabbit{ in the United States~}. Here is a description of it:

Example:

> The new species, called the American black rabbit, is native to the eastern and midwestern regions. It has a gray coat with black markings on its ears, legs and tail, and it has white markings on the back and sides of its ears and on the tips and tips of its ears and tail.The rabbit is a medium-sized mammal that weighs between 2 and 4 pounds (1.5 and 2 kilograms).

If you want to include any of the special characters in the prompt, you can escape them with `\`. There are also two other operators, ^ and /, which I explain below. The system allows operations to be combined arbitrarily, producing complex expressions like the following:

> {~Content warning: violence. }The following is a {true|faithful} \\& vivid account of what happened to me when I traveled to {Antarctica&the Arctic} to find the {{North&South} Pole|Yeti|meaning of life}.

The output looks like this:

> It is a story that is very close to my heart and I hope it will help others to find their own meaning in life.I was a 21 year old student at the time and had just completed my first year of university in the United States. I was living with my parents and was looking forward to my first year of graduate school in Canada, where I was studying to be a doctor.I had been living in the United Kingdom for the previous three years and had been to the North Pole on a previous trip. I had also visited the South pole and had a good feeling about it.I was planning to go to the South Pole again this year and had already purchased my plane ticket and all the necessary equipment. I was excited about my trip. I had a great time in Antarctica, and I had a lot of fun with my friends and family.On December 1, 1999, I boarded my plane for the last time in the United States and flew to the South Pole, arriving on December [...]

Below I explain the method and present some preliminary findings suggesting that this system can improve the performance of language models at certain tasks—by one benchmark, LAMBADA, even bringing relatively small models like the 774B parameter GPT-2 close to the reported performance of the massive GPT-3. It also gives the user new ways of controlling the output of text generators, opening the potential for more systematic, nuanced, and creative approaches to what is, at present, often a matter of trial and error.

## What It Does (and Why, and How)

This system was inspired in part by my research for my forthcoming book on the history of algorithms and, in particular, from a reconsideration of [George Boole](https://georgeboole.com/boole/)'s work on algebraic logic. The logic system that bears Boole's name (at least as it appears in programming languages like Python) is an arithmetic of two values, *true* and *false* or 1 and 0. This is not, however, how Boole's original system worked; his variables could have non-numeric values that represented classes of things such as "trees" and "sheep," and he used the *and* and *or* operators algebraically, not as operations to be executed. Like Boole's work, this system implements logical operators in a way that incorporates far more semantic content than the two-valued data type of standard programming languages. Instead of applying these operators to truth values, it applies them to English words that are to be fed into a language model.

While Boolean logic in its modern form is based primarily on *and*, *or*, and *not*, Boole's original logic system included a fourth operator that has largely been forgotten. If *or* is like addition, *not* subtraction, and *and* multiplication, then this operator is equivalent to division. Division, in Boole's system, is the inverse of *and*: dividing "small and fluffy" by "fluffy" gives us "small." He described division as the operation “by which from the conception of a given class of things we ascend to the conception of some larger class from which the given class would be formed from the mental selection of those individuals which possess a given property” (*Selected Manuscripts in Logic and Its Philosophy*, 58). This idea did not hold up in philosophical logic because there is not necessarily a unique category that meets this definition.

I hope to show that, even if it is problematic in logic, division does have a sensible meaning in regard to language models. Put simply, if subtraction works like "not prompt A," division works more like "a prompt that means not A." This is a highly useful effect, especially given the difficulties language models presently have dealing with negation, and it is the basis of the *more than* and *as opposed to* operators employed in this program.

This project also takes inspiration from some old insights from linguistic anthropology. Structuralists such as [Claude Lévi-Strauss](https://press.uchicago.edu/ucp/books/book/chicago/R/bo3614777.html) maintained that language is based on difference: in order to understand what it means for something to be raw, we also must understand the meaning of "cooked." Yet it is not always self-evident which words are opposites. Perhaps it is clear that the opposite of day is night, but what is the opposite of a narrative? Is it, as [Lev Manovich once posited](http://mfj-online.org/journalPages/MFJ34/Manovich_Database_FrameSet.html), a database? A scientific analysis? A photograph? Silence? The *as opposed to* operator makes it possible to specify which opposite one has in mind, thus guiding the generator with more precision.

### Syntax

My implementation of these operations is somewhat like a programming language, in that expressions are parsed into a syntax tree which is then compiled into a sequence of executable operations. The semantics, however, are very different compared to standard programming languages. Continuing my [longstanding interest](http://jeffreymbinder.net/208/homespring) in unusual programming languages, this project creates something of a hybrid between Boolean expressions and the English language. In executing programs, the system takes (and I nod here to [G. W. Leibniz](https://global.oup.com/academic/product/leibniz-dissertation-on-combinatorial-art-9780198837954?cc=us&lang=en&)) a combinatorial approach, generating all possible variants of the prompt based on the alternatives set out by the operators. It then runs the model on all of these variants and aggregates the resulting probability distributions based on the indicated operations.

For example, consider the following, very simple example:

> Hello|Goodbye

This expression provides two options: the text is either "Hello" or "Goodbye." In order to interpret it, the software first runs the model on each of these prompts. The result is two probability distributions over possible next tokens, which constitute the model's predictions for each prompt. These distributions are combined through an implementation of the *or* logic, producing a new distribution that is used in generation. As the program generates more words, it continues to consider both prompts so that the Boolean operator affects all of the output.

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

Note that the text to which the *as opposed to* operator applies—the word "They" as an opening for the second sentence—must still be considered as an option so that its effects can be discouraged. In order to combine the outputs, the software generates a simple program that indicates the order in which the operations must be performed:

> 0: 0 |= 1  
> 1: 2 |= 3  
> 2: 4 |= 5  
> 3: 2 ~= 4  
> 4: 0 &= 2  
> 5: 0 |= 6

This means that the values for prompts 0 and one are combined with the | operation and the result stored at position 0; the same is then done for 2 and 3, and so forth. In the end, the final result always ends up at position 0.

A downside of this method is the potential for combinatorial explosion—an exponential increase in the number of variants as the complexity of the prompt increases. This issue arises in cases where there are options in multiple parts of a single stretch of text, as in "{A|B}{C|D}{E|F}{G|H}{I|J}" (32 variants). If the operators are organized in a strictly nested structure, then the number of variants is simply equal to the number of nodes in the tree. Thus, the seemingly more complex expression "{{{A&B}|{C&{D~E}}}&{F|{{G&H}~{I^J}}}}" only produces 10 variants. Users will need to consider this when designing prompts.

All of the variants are fed into the model in one batch, so a prompt that involves more variants will not take much more time to run than a simple one, at least on a GPU. However, expressions will use up more GPU memory the more variants they involve.

### Semantics

The efficacy of this technique is strongly dependent on what calculations are used when applying the operators. Since there are a number of possible ways of applying Boolean logic to language models, I decided to outline several approaches I have tried and explain why I made the choices I did. My preferred approach is based on a combination of probability theory and logic, and it is specifically aimed at manipulating the semantic content of the prompts—that is, their meanings. This is a work in progress, and I welcome feedback and suggestions.

In all cases, we need to implement the rules for how *and* and *or* behave when applied to the multinomial probability distributions that text generation models output. A simple way of doing this would be to apply logical connectives to the prompts, considered as values of random variables. Suppose that Pr(gen = x | prompt = A) gives the probability that token x will be generated given prompt A. We could define the operators like so:

> Pr(gen = x | prompt = A and B) = Pr(gen = x | prompt = A and prompt = B)  
> Pr(gen = x | prompt = A or B) = Pr(gen = x | prompt = A or prompt = B)  
> Pr(gen = x | prompt = A and not B) = Pr(gen = x | prompt = A and prompt ≠ B)

However, this approach runs into conceptual problems because, in a standard text generation model, there can only be one prompt at a time. It therefore does not make sense to say that the prompt is both A and B, and employing this condition leads to inconsistent probabilities. As a result, this approach works for *or*, but it does not provide a sound way of defining *and*.

A better approach would be to apply the operators not to the prompts themselves, but rather to their (logical) meanings. While we cannot directly compute the effects of meaning on the text generator, we can come up with a formal system that approximates this effect. Suppose that exp is a Boolean prompt expression and and exp ⊃ A expresses "exp means A." We assume that prompts can have multiple meanings. We can interpret the *and* and *or* operators through the following rules:

> exp ⊃ A and B ⇒ exp ⊃ A and exp ⊃ B
> exp ⊃ A or B ⇒ exp ⊃ A or exp ⊃ B

We can then use the following rules for generating text:

> If A contains no operators, then Pr(gen = x | exp ⊃ A) = Pr(gen = x | prompt = A)  
> Pr(gen = x | exp ⊃ A and exp ⊃ B) = Pr(gen1 = x | gen1 = gen2, exp1 ⊃ A, exp2 ⊃ B)

Note that the last formula involves two separate generative processes, one for each prompt. We assume that these processes operate independently. Based on these definitions, we can derive the following:

> Pr(gen = x | exp ⊃ A and exp ⊃ B) = Pr(gen = x | exp ⊃ A) Pr(gen = x | exp ⊃ B) / Pr(gen1 = gen2 | exp1 ⊃ A and exp2 ⊃ B)  
> Pr(gen = x | exp ⊃ A or exp ⊃ B) = (Pr(gen = x | exp ⊃ A) Pr(exp ⊃ A) + Pr(gen = x | exp ⊃ B) Pr(exp ⊃ B) - Pr(gen = x | exp ⊃ A and exp ⊃ B) Pr(exp ⊃ A and exp ⊃ B)) / (Pr(exp ⊃ A) + Pr(exp ⊃ B) - Pr(exp ⊃ A and exp ⊃ B))

The formula for *and* requires computing the probability for gen1 = gen2, which is simply the dot product of the two prediction vectors. For *or*, we need probabilities for A and B and for their co-occurrence. The probabilities for A and B affect the relative weights assigned to them, which the program currently sets to be equal. Pr(exp ⊃ A and exp ⊃ B) may be adjusted so as to encode different assumptions about how much overlap occurs between the meanings of prompts; it can be controlled using the `overlap_factor` parameter, which is generally best set to around 0.25. This is the method used for the the & and | operators.

For *not*, the situation is less straightforward. One potential approach would be the following:

> exp ⊃ A and not B ⇒ exp ⊃ A and exp ⊅ B

That is, we interpret "A and not B" as an expression that means A and does not mean B. We can then derive the following formula for predictions:

> Pr(gen = x | exp ⊅ B) = (Pr(gen = x) - Pr(gen = x | exp ⊃ B) Pr(exp ⊃ B)) / (1 - Pr(exp ⊃ B))

In order to use this formula, we would need a value for Pr(gen2 = x), which indicates the probability that token x will be generated given no information whatsoever about the prompt. A reasonable approximation of this would be a uniform distribution, which has the advantage of introducing no particular bias into the results. This leads to the following approximation:

> Pr(gen = x | exp ⊃ A and not B) ∝ Pr(gen = x | exp ⊃ A) (1 - k Pr(gen = x | exp ⊃ B))

Where k can be set to the highest value that does not produce a negative probability. This form of negation is modestly effective at suppressing certain words, but it comes at the cost of decreasing the coherence of the generated text. You can try this version of "A and not B" using the ^ operator, which is mainly of theoretical interest.

Thankfully, negation also admits another interpretation that works better in practice. What if, instead of interpreting negation as "exp does not mean B," we interpret it as "exp means 'not B'"? That is:

> exp ⊃ A and not B ⇒ exp ⊃ A and exp ⊃ not B

This is not logically equivalent to the above, since denying that an expression means B does not imply that the expression has the meaning "not B"; the expression may, instead, have nothing to say about B either way. This thinking can lead us to a different sort of negation operator, although justifying it requires a bit more work.

My current approach is based on considering what should happen when we construct "B and not B." An ancient logical principle is [*ex contradictione sequitur quodlibet*](https://en.wikipedia.org/wiki/Principle_of_explosion) (from contradiction, anything follows). That is, from the proposition "B and not B," anything whatsoever may be inferred. If we think of generated text as involving inferences from the meaning of the prompt, we might suppose that a prompt meaning both "B" and "not B" should lead the model to generate text with no particular relation to B. That is, we want the following to hold:

> Pr(gen = x | exp ⊃ B and not B) = Pr(gen = x)

Based on this, one can readily prove that, given the definition we have chosen for *and*, the predictions for "not B" must have the following form:

> Pr(gen = x | exp ⊃ not B) ∝ Pr(gen = x) / Pr(gen = x | exp ⊃ B)

The value of Pr(gen = x) indicates the broader distribution from which B is to be removed. One option is to set it, once again, to a uniform distribution, thus making no assumptions about the range of tokens that may be generated. In practice, however, it typically makes sense to employ a more targeted value. In particular, it is useful to set set Pr(gen = x) to Pr(gen = x | exp ⊃ A) for some other prompt A. Under this assumption, the predictions for "not B" have the following form:

> Pr(gen = x | exp ⊃ not B) ∝ Pr(gen = x | exp ⊃ A) / Pr(gen = x | exp ⊃ B)

I have made this calculation available in the program with the / operator. In effect, this operator causes the program to choose words that are more probable with A than with B (hence the name *more than*). It is equivalent to "A and not B" with a uniform distribution for Pr(gen = x).

The effect of / is only useful in combination with other operators, since by itself "A/B" produces nonsense. The most obvious way of doing this is to construct an expression of the form "A and not B," with the negation alternative set to A. This gives us the following:

> Pr(gen = x | exp ⊃ A and not B) ∝ Pr(gen = x | exp ⊃ A)^2 / Pr(gen = x | exp ⊃ B)

This is the calculation used for "A~B"; it is equivalent to "A&{A/B}." Put simply, it generates text using prompt A while asserting that the prompt means "A and not B." I have found that this method is effective not just at discouraging generators from doing certain things, but also at improving their performance when applied to certain tasks.

There may also be some use in expressions of the form "A&{B/C}," which generates text using A while biasing the output in favor of words that are more probable with B than with C. For instance, one may use "adorable, cute kitty/cat" to inject cuteness into descriptions of any animal:

> Scientists recently discovered a new species of {snake&{adorable, cute kitty/cat}}. Here is a description of it:

> This is the most adorable snake I've seen. It's a little guy with an orange belly and white spots on its head. He has two tiny, black spots on his back. He has two little black dots on his sides and his eyes are a bright orange. His mouth is a bit bigger than his body. He is very small. I love him. I'm not sure how to pronounce his name. I'm calling him "Buddy".

### Implementation

The main logic of this system, including the compiler and code for executing programs, appears in `program.py`. The file `generation_utils.py` amends the corresponding file from the Transformers package to invoke this system. The easiest way to get started is to run `generate.py`, whose source can be edited to change the settings. The system works in basically the same way as the regular Transformers generator, but instead of passing a sequence of token ids to `generate()`, you pass in a string containing the Boolean expression. To run this code, you will need to have recent Git master versions of PyTorch and Transformers installed.

The code in this repo implements Boolean prompting for autoregressive text generators; it can be used with GPT, GPT2, XLNet, XLM, CTRL, and Transformer XL models, either with or without finetuning. The broader technique I describe is not, however, specific to this type of generator. In my project [A Hundred Visions and Revisions](https://github.com/jeffbinder/visions-and-revisions), first published in March 2020, I incorporated a technique similar to the *as opposed to* operator (there called "strong topic bias") into a non-autoregressive text rewriting procedure based on [BERT](https://github.com/google-research/bert)-style masked language models.

## Experiments

I am continuing to develop and experiment with this method, but I do have some preliminary results.

### Discouraging Words

An obvious application of this system is to prevent (or at least discourage) the model from doing certain things. Evaluating success at this goal is difficult, since judgments about the quality and meaning of texts can be debatable. However, it is possible to test the effect in a rough way by checking how often certain, generally undesirable words appear.

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

The *as opposed to* operator provides an alternative way of specifying negation that produces much better results:

> Prompt C. Scientists recently discovered a new species of snake{~ with legs}. Here is a description of it:

The results are as follows:

| Words | Prompt A | Prompt C | p A\>C |
| --- | --- | --- | --- |
| leg/legs/legged | 112/1000 | 37/1000 | ~9e-11 |

As the results show, the method does indeed significantly reduce the references to legs in the output.

This technique is not the only way the *as opposed to* operator may be applied. Another approach is to utilize both sides of the operator, indicating both what the animal is and what it is not. This technique can be used to discourage those irksome references to hair and fur:

> Prompt D. Scientists recently discovered a new species of {snake~mammal}. Here is a description of it:

These are the results:

| Words | Prompt A | Prompt D | p A\>D |
| --- | --- | --- | --- |
| leg/legs/legged | 97/1000 | 99/1000 | N/A |
| fur/furred/furry | 17/1000 | 6/1000 | ~0.01 |
| hair/hairs/haired/hairy | 63/1000 | 42/1000 | ~0.02 |

As these results show, adding the operator to the prompt decreased the incidence of words referring to the mammalian traits of fur and (to a lesser extent) hair. It also results in a somewhat smaller quantity of legs.

A potential application of the *as opposed to* operator would be in discouraging the model from generating text with offensive, violent, or otherwise undesirable qualities. As such, this method would be conceptually similar to the ["self-debiasing"](https://arxiv.org/pdf/2103.00453.pdf) method proposed by Schick, Udupa, and Schütze. I would caution, however, that the technique's ability to discourage offensive text is only as good as the model's ability to distinguish offensive from non-offensive, which is to say (at least in the case of GPT-2) not that great. Such efforts also run into political difficulties: people do not all agree on what is offensive, and it is a complex matter to determine exactly what stance on this issue has wound up embedded in the model. Thus, while putting something like "{~Content warning: racism. }" into a prompt might mitigate the problem to some extent, it should not be taken as a solution.

As a means of controlling text generators, this technique is probably not as effective as techniques like [GeDi](https://github.com/salesforce/GeDi), which uses a classification model to suppress undesired qualities. What is interesting about this use of Boolean prompting is that it enables users to describe what they want to discourage in natural language, which is interpreted using nothing but a pretrained model.

### LAMBADA

The [LAMBADA](https://zenodo.org/record/2630551#.YWb8Iy-cbOQ) benchmark tests the ability of models to account for long-range dependencies in texts. LAMBADA is based on a collection of excerpts from novels, selected such that humans are able to guess the final word, but only when they are given the whole passage. The model is supplied with the passage absent the final word, and must predict what that word is.

The LAMBADA benchmark is usally scored in two ways, the accuracy of predictions (measured as a percentage) and perplexity, which measures how well the probabilities produced by the model align with a text. Since the present technique alters the way the continuation is generated without altering the model itself, perplexity is not clearly applicable, so I have focused solely on accuracy.

The creators of GPT-2 [report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) that the largest version of their model attained an accuracy of **63.24**. Accoring to [this discussion](https://github.com/openai/gpt-2/issues/131), this experiment was based on predicting the final token of the text, not the final word. Since GPT-2 sometimes splits words into multiple tokens, this is an easier task than predicting whole words. I evaluated my method with both versions of the task to enable fair comparisons with other results.

The performance of GPT-3 was tested using several different approaches. The authors advocate a "few-shot" approach, in which the model is primed with several completed examples before being presented with a problem; the [reported accuracy](https://arxiv.org/abs/2005.14165v4) with this approach is **86.4**. This approach is not, however, generally workable with smaller models, so I did not attempt to replicate it here. GPT-3's reported accuracy with the zero-shot approach, which I have employed in this experiment, is **76.2**.

I tried several formulae for constructing the prompt. Here "context" refers to the passage, excluding the final word or token that is to be predicted.

1. context  
2. context~  
3. context~[...]last word of context  
4. context~[...]last phrase of context  
5. context~[...]last sentence of context  
6. context~[...]{last sentence|last word}  
7. context~[...]{last sentence|last phrase}

The rationale is to discourage the model from making predictions based solely on the later parts of the prompt while ignoring the earlier ones. Phrases are delineated using the following regex: `[,.:;?!"“”]`. Sentence boundaries are determined using the Punkt sentence tokenizer.

These are the results for token prediction:

| Model | # params | Baseline | Blank | Last word | Last phrase | Last sentence | Last sentence or word | Last sentence or phrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 117M | 48.03 | 58.63 | 62.37 | 65.61 | 67.71 | 67.81 | 67.73 |
| gpt2-medium | 345M | 56.76 | 61.50 | 70.66 | 72.39 | 74.48 | 74.29 | 74.50 |
| gpt2-large | 774M | 60.95 | 67.86 | 74.13 | 75.65 | 77.45 | 77.47 | 77.35 |
| gpt2-xl | 1558M | 63.98 | 70.17 | 76.38 | 77.47 | 78.83 | 79.22 | 78.96 |

These are the results for whole-word prediction:

| Model | # params | Baseline | Blank | Last word | Last phrase | Last sentence | Last sentence or word | Last sentence or phrase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | 117M | 34.10 | 45.33 | 50.73 | 54.98 | 58.16 | 58.30 | 58.10 |
| gpt2-medium | 345M | 44.79 | 47.72 | 62.10 | 63.46 | 66.93 | 66.99 | 67.07 |
| gpt2-large | 774M | 50.05 | 56.61 | 66.02 | 67.79 | 70.97 | 70.79 | 70.68 |
| gpt2-xl | 1558M | 53.87 | 59.13 | 68.60 | 69.90 | 72.39 | 72.87 | 72.54 |

You can replicate these results using the `lambada_score.py` script. Note that the use of the | operator only works when the overlap factor is set to 0.

It is worth noting that the scores are only improved with the *as opposed to* operator, as implemented using division; the ^ operator based on subtraction does not work in this application.

To some extent, these techniques work by exploiting the specific nature of the test dataset. LAMBADA was designed for testing a model's ability to find relevant information that occurs in previous sentences. By boosting the effects of the earlier parts of the prompt, the negation operator ensures that the model considers the prompt as a whole.

From an AI perspective, this approach might be seen as a way of gaming the system. as opposed to improving the model, the program is sifting through the output to find the predictions that best suit the nature of the LAMBADA test. Yet this sifting is only a problem if we insist that the machine be capable of both determining the nature of the task and performing it without human intervention. If we see language models not as [a foundation for general intelligence](https://arxiv.org/pdf/2108.07258.pdf), but rather as a practical means of performing computations, then designing prompt expressions that suit the task at hand is a useful technique. Boolean prompting is an alternative to AI purism, an approach that enables human and machine to work together.

There is also reason to think that at least some of the improvement stems from something other than the specific nature of the task. For all model sizes except medium, simply adding a ~ operator to the end of the input improved the performance by more than five percentage points. This intervention makes no assumptions about the task at hand; it merely encodes the prior that the prompt contains some information that is relevant to the generative task. As an explanation of the increased performance, I would hypothesize that the model is conflating the overall frequency of a word with its probability in a particular context; the *as opposed to* operator induces the system to focus more on the context. This technique could potentially be applied to any number of tasks.
