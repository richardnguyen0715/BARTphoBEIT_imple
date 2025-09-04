from BARTphoBEIT import *

def predict_single_question(model_path, image_path, question, config):
    """Predict answer for a single question"""
    
    # Load model
    model = VietnameseVQAModel(config)
    model.load_state_dict(torch.load(model_path))
    model = model.to(config['device'])
    model.eval()
    
    # Prepare input
    tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    feature_extractor = ViTFeatureExtractor.from_pretrained(config['vision_model'])
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = image_inputs['pixel_values'].to(config['device'])
    
    # Process question
    question_encoding = tokenizer(
        question,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    question_input_ids = question_encoding['input_ids'].to(config['device'])
    question_attention_mask = question_encoding['attention_mask'].to(config['device'])
    
    # Generate answer
    with torch.no_grad():
        generated_ids = model(
            pixel_values=pixel_values,
            question_input_ids=question_input_ids,
            question_attention_mask=question_attention_mask
        )
    
    answer = model.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return answer