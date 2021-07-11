## Subpopulation

`Subpopulation` is to identify the specific part of dataset on which the target model performs poorly. To retrieve a subset that meets the configuration, `Subpopulation` divides the dataset through sorting samples by certain attributes. We also support the following `Subpopulation`:

<table style="width:100%" border="2">
<thead>
  <tr>
    <th>Subpopulation</th>
    <th>Description</th>
    <th>Reference</th>
  </tr>
</thead>
<tbody>
  <tr>
      <td><code>LMSubPopulation_0%-20%</code></td>
      <td><sub>Filter samples based on the text perplexity from a language model (i.e., GPT-2), 0-20% is the lower part of the scores.</sub></td>
      <td rowspan="8"><sub>Robustness Gym: Unifying the NLP Evaluation Landscape (https://arxiv.org/pdf/2101.04840)</sub></td>
  </tr>
  <tr>
      <td><code>LMSubPopulation_80%-100%</code></td>
      <td><sub>Filter samples based on the text perplexity from a language model (i.e., GPT-2), 80-100% is the higher part of the scores.</sub></td>
  </tr>
  <tr>
      <td><code>LengthSubPopulation_0%-20%</code></td>
    <td><sub>Filter samples based on text length, 0-20% is the lower part of the length.</sub></td>
  </tr>
  <tr>
      <td><code>LengthSubPopulation_80%-100%</code></td>
      <td><sub>Filter samples based on text length, 80-100% is the higher part of the length.</sub></td>
  </tr>
  <tr>
      <td><code>PhraseSubPopulation-negation</code></td>
    <td><sub>Filter samples based on a group of phrases, the remaining samples contain negation words (e.g., not, don't, aren't, no).</sub></td>
  </tr>
  <tr>
      <td><code>PhraseSubPopulation-question</code></td>
      <td><sub>Filter samples based on a group of phrases, the remaining samples contain question words (e.g., what, which, how, when).</sub></td>
  </tr>
  <tr>
      <td><code>PrejudiceSubpopulation-man</code></td>
      <td><sub>Filter samples based on gender bias, the chosen samples only contain words related to male (e.g., he, his, father, boy).</sub></td>
  </tr>
  <tr>
      <td><code>PrejudiceSubpopulation-woman</code></td>
    <td><sub>Filter samples based on gender bias, the chosen samples only contain words related to female (e.g., she, her, mother, girl)</sub></td>
  </tr>
</tbody>
</table>

In addition, you can define your own subpopulation follow this [tutorial](2_SubPopulation.ipynb).