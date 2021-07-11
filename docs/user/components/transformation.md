## Transformation

In order to verify the robustness comprehensively, TextFlint offers 20 universal transformations and 60 task-specific transformations, covering 12 NLP tasks. The following table summarizes the `Transformation` currently supported and the examples for each transformation can be found in our [website](https://www.textflint.com).

<div style="overflow-x: auto; overflow-y: auto; height: 1000px; width:100%;">
<table style="width:1000px" border="2">
<thead>
  <tr>
    <th>Task</th>
    <th>Transformation</th>
    <th>Description</th>
    <th>Reference</th>
  </tr>
</thead>
<tbody >
  <tr>
    <td rowspan="20">UT (Universal Transformation)</td>
      <td><code>AppendIrr</code></td>
      <td><sub>Extend sentences by irrelevant sentences</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>BackTrans</code></td>
      <td><sub>BackTrans (Trans short for translation) replaces test data with paraphrases by leveraging back translation, which is able to figure out whether or not the target models merely capture the literal features instead of semantic meaning. </sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>Contraction</code></td>
      <td><sub>Contraction replaces phrases like `will not` and `he has` with contracted forms, namely, `won’t` and `he’s`</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>InsertAdv</code></td>
    <td><sub>Transforms an input by add adverb word before verb</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Keyboard</code></td>
      <td><sub>Keyboard turn to the way how people type words and change tokens into mistaken ones with errors caused by the use of keyboard, like `word → worf` and `ambiguous → amviguius`.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>MLMSuggestion</code></td>
    <td><sub>MLMSuggestion (MLM short for masked language model) generates new sentences where one syntactic category element of the original sentence is replaced by what is predicted by masked language models.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Ocr</code></td>
    <td><sub>Transformation that simulate ocr error by random values.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Prejudice</code></td>
    <td><sub>Transforms an input by Reverse gender or place names in sentences.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Punctuation</code></td>
    <td><sub>Transforms input by add punctuation at the end of sentence.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>ReverseNeg</code></td>
    <td><sub>Transforms an affirmative sentence into a negative sentence, or vice versa.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SpellingError</code></td>
    <td><sub>Transformation that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.</sub></td>
    <td><sub>Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs (https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf)</sub></td>
  </tr>
  <tr>
    <td><code>SwapAntWordNet</code></td>
    <td><sub>Transforms an input by replacing its words with antonym provided by WordNet.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapNamedEnt</code></td>
    <td><sub>Swap entities with other entities of the same category. </sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapNum</code></td>
    <td><sub>Transforms an input by replacing the numbers in it.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapSynWordEmbedding</code></td>
    <td><sub>Transforms an input by replacing its words by Glove.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapSynWordNet</code></td>
    <td><sub>Transforms an input by replacing its words with synonyms provided by WordNet. </sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Tense</code></td>
    <td><sub>Transforms all verb tenses in sentence.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>TwitterType</code></td>
    <td><sub>Transforms input by common abbreviations in TwitterType.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Typos</code></td>
      <td><sub>Randomly inserts, deletes, swaps or replaces a single letter within one word (Ireland → Irland).</sub></td>
      <td><sub>Synthetic and noise both break neural machine translation (https://arxiv.org/pdf/1711.02173.pdf)</sub></td>
  </tr>
  <tr>
    <td><code>WordCase</code></td>
      <td><sub>Transform an input to upper and lower case or capitalize case.</sub></td>
    <td>-</td>
  </tr>    
  <tr>
    <td rowspan="7">RE (Relation Extraction)</td>
      <td><code>InsertClause</code></td>
      <td><sub>InsertClause is a transformation method which inserts entity description for head and tail entity</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapEnt-LowFreq</code></td>
      <td><sub>SwapEnt-LowFreq is a sub-transformation method from EntitySwap which replace entities in text with random same typed entities with low frequency.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapTriplePos-Birth</code></td>
      <td><sub>SwapTriplePos-Birth is a transformation method specially designed for birth relation. It paraphrases the sentence and keeps the original birth relation between the entity pairs.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapTriplePos-Employee</code></td>
      <td><sub>SwapTriplePos-Employee is a transformation method specially designed for employee relation. It deletes the TITLE description of each employee and keeps the original employee relation between the entity pairs.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapEnt-SamEtype</code></td>
      <td><sub>SwapEnt-SamEtype is a sub-transformation method from EntitySwap which replace entities in text with random entities with the same type.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapTriplePos-Age</code></td>
      <td><sub>SwapTriplePos-Age is a transformation method specially designed for age relation. It paraphrases the sentence and keeps the original age relation between the entity pairs.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapEnt-MultiType</code></td>
      <td><sub>SwapEnt-MultiType is a sub-transformation method from EntitySwap which replace entities in text with random same-typed entities with multiple possible types.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">NER (Named Entity Recognition)</td>
      <td><code>EntTypos</code></td>
      <td><sub>Swap/delete/add random character for entities</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>ConcatSent</code></td>
      <td><sub>Concatenate sentences to a longer one.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>SwapLonger</code></td>
      <td><sub>Substitute short entities to longer ones</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>CrossCategory</code></td>
      <td><sub>Entity Swap by swaping entities with ones that can be labeled by different labels.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>OOV</code></td>
    <td><sub>Entity Swap by OOV entities.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">POS (Part-of-Speech  Tagging)</td>
    <td><code>SwapMultiPOSRB</code></td>
    <td><sub>It is implied by the phenomenon of conversion that some words hold multiple parts of speech. That is to say, these multi-part-of-speech words might confuse the language models in terms of POS tagging. Accordingly, we replace adverbs with words holding multiple parts of speech.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapPrefix</code></td>
    <td><sub>Swapping the prefix of one word and keeping its part of speech tag.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapMultiPOSVB</code></td>
    <td><sub>It is implied by the phenomenon of conversion that some words hold multiple parts of speech. That is to say, these multi-part-of-speech words might confuse the language models in terms of POS tagging. Accordingly, we replace verbs with words holding multiple parts of speech.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapMultiPOSNN</code></td>
    <td><sub>It is implied by the phenomenon of conversion that some words hold multiple parts of speech. That is to say, these multi-part-of-speech words might confuse the language models in terms of POS tagging. Accordingly, we replace nouns with words holding multiple parts of speech.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapMultiPOSJJ</code></td>
    <td><sub>It is implied by the phenomenon of conversion that some words hold multiple parts of speech. That is to say, these multi-part-of-speech words might confuse the language models in terms of POS tagging. Accordingly, we replace adjectives with words holding multiple parts of speech.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="6">COREF (Coreference Resolution)</td>
    <td><code>RndConcat</code></td>
    <td><sub>RndConcat is a task-specific transformation of coreference resolution, this transformation will randomly retrieve an irrelevant paragraph from the corpus, and concatenate it after the original document</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>RndDelete</code></td>
    <td><sub>RndDelete is a task-specific transformation of coreference resolution, through this transformation, there is a possibility (20% by default) for each sentence in the original document to be deleted, and at least one sentence will be deleted; related coreference labels will also be deleted</sub></td>
     <td>-</td>
    </tr>
  <tr>
    <td><code>RndReplace</code></td>
    <td><sub>RndInsert is a task-specific transformation of coreference resolution, this transformation will randomly retrieve irrelevant sentences from the corpus, and replace sentences from the original document with them （the proportion of replaced sentences and original sentences is 20% by default）</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>RndShuffle</code></td>
    <td><sub>RndShuffle is a task-specific transformation of coreference resolution, during this transformation, a certain number of swapping will be processed, which swap the order of two adjacent sentences of the original document （the number of swapping is 20% of the number of original sentences by default）</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>RndInsert</code></td>
    <td><sub>RndInsert is a task-specific transformation of coreference resolution, this transformation will randomly retrieve irrelevant sentences from the corpus, and insert them into the original document （the proportion of inserted sentences and original sentences is 20% by default）</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>RndRepeat</code></td>
    <td><sub>RndRepeat is a task-specific transformation of coreference resolution, this transformation will randomly pick sentences from the original document, and insert them somewhere else in the document （the proportion of inserted sentences and original sentences is 20% by default）</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="3">ABSA (Aspect-based Sentiment Analysis)</td>
    <td><code>RevTgt</code></td>
    <td><sub>RevTgt: reverse the sentiment of the target aspect.</sub></td>
      <td rowspan="3"><sub>Tasty Burgers, Soggy Fries: Probing Aspect Robustness in Aspect-Based Sentiment Analysis (https://www.aclweb.org/anthology/2020.emnlp-main.292.pdf)</sub></td>
  </tr>
  <tr>
    <td><code>AddDiff</code></td>
    <td><sub>RevNon: Reverse the sentiment of the non-target aspects with originally the same sentiment as target.</sub></td>
  </tr>
  <tr>
    <td><code>RevNon</code></td>
    <td><sub>AddDiff: Add aspects with the opposite sentiment from the target aspect.</sub></td>
  </tr>
  <tr>
    <td rowspan="5">CWS (Chinese Word Segmentation)</td>
    <td><code>SwapContraction</code></td>
    <td><sub>SwapContriction is a task-specific transformation of Chinese Word Segmentation, this transformation will replace some common abbreviations in the sentence with complete words with the same meaning</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapNum</code></td>
    <td><sub>SwapNum is a task-specific transformation of Chinese Word Segmentation, this transformation will replace the numerals in the sentence with other numerals of similar size</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapSyn</code></td>
    <td><sub>SwapSyn is a task-specific transformation of Chinese Word Segmentation, this transformation will replace some words in the sentence with some very similar words</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapName</code></td>
    <td><sub>SwapName is a task-specific transformation of Chinese Word Segmentation, this transformation will replace the last name or first name of the person in the sentence to produce some local ambiguity that has nothing to do with the sentence</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapVerb</code></td>
    <td><sub>SwapName is a task-specific transformation of Chinese Word Segmentation, this transformation will transform some of the verbs in the sentence to other forms in Chinese</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="3">SM (Semantic Matching)</td>
    <td><code>SwapWord</code></td>
    <td><sub>This transformation will add some meaningless sentence to premise, which do not change the semantics.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapNum</code></td>
    <td><sub>This transformation will find some num words in sentences and replace them with different num word.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>Overlap</code></td>
    <td><sub>This method generate some data by some template, whose hypotheis and sentence1 have many overlap but different meaning.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">SA (Sentiment Analysis)</td>
    <td><code>SwapSpecialEnt-Person</code></td>
    <td><sub>SpecialEntityReplace-Person is a task-specific transformation of sentiment analysis, this transformation will identify some special person name in the sentence, randomly replace it with other entity names of the same kind</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>SwapSpecialEnt-Movie</code></td>
    <td><sub>SpecialEntityReplace is a task-specific transformation of sentiment analysis, this transformation will identify some special movie name in the sentence, randomly replace it with other movie name.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>AddSum-Movie</code></td>
    <td><sub>AddSummary-Movie is a task-specific transformation of sentiment analysis, this transformation will identify some special movie name in the sentence, and insert the summary of these entities after them (the summary content is from wikipedia).</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>AddSum-Person</code></td>
    <td><sub>AddSummary-Person is a task-specific transformation of sentiment analysis, this transformation will identify some special person name in the sentence, and insert the summary of these entities after them (the summary content is from wikipedia).</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>DoubleDenial</code></td>
    <td><sub>SpecialWordDoubleDenial is a task-specific transformation of sentiment analysis, this transformation will find some special words in the sentence and replace them with double negation</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="4">NLI (Natural Language Inference)</td>
    <td><code>NumWord</code></td>
    <td><sub>This transformation will find some num words in sentences and replace them with different num word.</sub></td>
    <td rowspan="3"><sub>Stress Test Evaluation for Natural Language Inference (https://www.aclweb.org/anthology/C18-1198/)</sub></td>
  </tr>
  <tr>
    <td><code>SwapAnt</code></td>
    <td><sub>This transformation will find some keywords in sentences and replace them with their antonym.</sub></td>
  </tr>
  <tr>
    <td><code>AddSent</code></td>
    <td><sub>This transformation will add some meaningless sentence to premise, which do not change the semantics.</sub></td>
  </tr>
  <tr>
    <td><code>Overlap</code></td>
    <td><sub>This method generate some data by some template, whose hypotheis and premise have many overlap but different meaning.</sub></td>
    <td><sub>Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference (https://www.aclweb.org/anthology/P19-1334/)</sub></td>
  </tr>
  <tr>
    <td rowspan="5">MRC (Machine Reading Comprehension)</td>
    <td><code>PerturbQuestion-MLM</code></td>
    <td><sub>PerturbQuestion is a task-specific transformation of machine reading comprehension, this transformation paraphrases the question.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>PerturbQuestion-BackTrans</code></td>
    <td><sub>PerturbQuestion is a task-specific transformation of machine reading comprehension, this transformation paraphrases the question.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td><code>AddSentDiverse</code></td>
    <td><sub>AddSentenceDiverse is a task-specific transformation of machine reading comprehension, this transformation generates a distractor with altered question and fake answer.</sub></td>
    <td rowspan="2"><sub>Adversarial Augmentation Policy Search for Domain and Cross-LingualGeneralization in Reading Comprehension (https://arxiv.org/pdf/2004.06076)</sub></td>
  </tr>
      <tr>
    <td><code>PerturbAnswer</code></td>
    <td><sub>PerturbAnswer is a task-specific transformation of machine reading comprehension, this transformation transforms the sentence with golden answer based on specific rules.</sub></td>
  </tr>
  <tr>
    <td><code>ModifyPos</code></td>
    <td><sub>ModifyPosition is a task-specific transformation of machine reading comprehension, this transformation rotates the sentences of context.</sub></td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="2">DP (Dependency Parsing)</td>
      <td><code>AddSubtree</code></td>
    <td><sub>AddSubtree is a task-specific transformation of dependency parsing, this transformation will transform the input sentence by adding a subordinate clause from WikiData.</sub></td>
    <td>-</td>
  </tr>
  <tr>
      <td><code>RemoveSubtree</code></td>
      <td><sub>RemoveSubtree is a task-specific transformation of dependency parsing, this transformation will transform the input sentence by removing a subordinate clause.</sub></td>
    <td>-</td>
  </tr>
</tbody>
</table>
</div>

In addition, you can define your own Transformation follow this [tutorial](1_Transformaions.ipynb). We also provide a [Interactive Demo](https://www.textflint.com/tutorials) to show how TextFlint can perform transformations on different tasks.
