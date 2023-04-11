import torch
from dataloader import get_data_loader

def load_model(device):
    beit_model = torch.load('data/model.pth', map_location=device)
    model = beit_model
    model.to(device)
    model.eval()
    return model

def setup_device():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f'using {device}')
    return device

def main():
    device = setup_device()
    model = load_model(device)
    data_loader = get_data_loader()

    predictions = []
    def eval_batch(model, label2ans, image, language_tokens, padding_mask, labels=None, qid=None):
        logits = model(
            image=image, question=language_tokens, 
            padding_mask=padding_mask)
        _, preds = logits.max(-1)
        for image_id, pred in zip(qid, preds):
            predictions.append({
                "question_id": image_id.item(), 
                "answer":  label2ans[pred.item()], 
            })

    with torch.no_grad():
        for data in data_loader:
            for tensor_key in data.keys():
                data[tensor_key] = torch.stack([data[tensor_key][0]]).to(device, non_blocking=False)

            if device.type=='cuda':
                with torch.cuda.amp.autocast():
                    eval_batch(model=model, label2ans=data_loader.dataset.label2ans, **data)
            else:
                eval_batch(model=model, label2ans=data_loader.dataset.label2ans, **data)

    print(f'Num predictions: {len(predictions)}, First prediction: {predictions[0]}')

if __name__=='__main__':
    main()