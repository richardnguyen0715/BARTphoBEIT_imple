import pandas as pd
import ast

# Load dataframe
df = pd.read_csv('your_vqa_dataset.csv')

# Sử dụng hàm prepare_data
def prepare_data(df):
    """Prepare data for evaluation"""
    # Parse answers if they're string representations of lists
    df = df[~df['question'].str.startswith("Mối quan hệ")]
    print(f"Dataframe len: {len(df)}")
    
    if isinstance(df['answers'].iloc[0], str):
        try:
            df['answers'] = df['answers'].apply(ast.literal_eval)
        except Exception:
            # fallback: wrap as single-element list
            df['answers'] = df['answers'].apply(lambda x: [x])
    
    # Create questions list
    questions = []
    for idx, row in df.iterrows():
        questions.append({
            'question_id': row.get('index', idx),
            'image_name': row['image_name'],
            'question': row['question'],
            'answers': row['answers'],
            'ground_truth': row['answers'][0] if row['answers'] else ""
        })
    
    print(f"Loaded {len(questions)} questions from dataset")
    return questions