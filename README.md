
# Adversarial Attack on LLM Model

## File Structure

### Main File
- **`\TURL\adversarial_attacks.ipynb`**: This is the main file for the adversarial attack. It includes:
  - Using the pre-trained model `bert-base-uncased` located in `\TURL\data\pre-trained_models\` to predict tables inside `\TURL\data\wikitables_v2\test.all_table_col_type.json`.
  - Ranking the logits differences of each entity in a given column of a given table.
  - Applying different strategies to attack the LLM.

### Evaluation Results
- **`\TURL\data\Entity_swapping.xlsx`**: This file contains the accuracy, precision, recall, and F1 scores for different attacking strategies.

### Files in **`\TURL\data`**

#### **`\wikitables_v2`**
- **`processed_WikiCT\test.pickle`**: Processed tables for prediction.
- **`classified_entities.txt`**: Classified entities for each of the 255 column types.
- **`entity_classes.py`**: Classified column types, used for swapping entities of the same category.
- **`entity_vocab.txt`**: All the entities.
- **`get_all_entities.py`**: Script for parsing `train.table_col_type.json` to output `classified_entities.txt`.
- **`test.all_table_col_type.json`**: All the tables in the original version of `test.table_col_type.json`.
- **`test.table_col_type.json`**: Tables used for prediction.
- **`train.table_col_type.json`**: Tables used for training.
- **`type_vocab.txt`**: All the column types.

#### **`\logits_difference`**
- This folder contains one CSV file for each table in `test.all_table_col_type.json`. Each CSV file has the logits differences obtained by the original logits of all the entities in the given table. 
  - Logit difference of entity A = Original logit of the correct column type - Logit of the correct column type after masking entity A.

