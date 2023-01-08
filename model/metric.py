import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def mlc_accuracy(output,target): # multi label classification accuracy 
    correct = 0
    # 수정
    for i in range(target.shape[0]):
        output[i] = torch.tensor([1 if x > 0.5 else 0 for x in output[i]])
        if torch.equal(output[i],target[i]):
            correct+=1
    
    return correct / target.shape[0]


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)