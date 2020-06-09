import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import time
import copy
from tqdm import tqdm
import numpy as np
import torchvision
   
from tensorboard_utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import cohen_kappa_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from config import INPUT_SIZE

import gc

def plotify(train_losses, val_losses, kappa_scores, auc_roc_scores):

    plt.plot(train_losses, label = 'Train Loss', color='orange')
    plt.plot(val_losses, label = 'Test Loss', color='blue')
    plt.plot(kappa_scores, label = 'Kappa score', color='black')
    plt.plot(auc_roc_scores, label = 'AUC ROC score', color='green')
    plt.show()


def train(criterion, model, optimizer, n_epochs, device, train_loader, val_loader, writer, scheduler):    
    since = time.time()

    train_losses = []
    valid_losses = []
    kappa_scores = []
    auc_roc_scores = []
    accuracy_scores = []
    valid_loss_min = np.Inf

    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    
    running_loss = 0
    for epoch in tqdm(range(1, n_epochs+1)):
        print ('\n')
        print ("="*30)
        print (f'\nEpoch : {epoch}') 
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            labels = labels.view(-1,1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels, cat='train')
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*images.size(0)
            
            running_loss += loss.item()
            update_step = 500
            if batch_idx % update_step == update_step-1:#999:    # every 1000 mini-batches...
                # ...log the running loss
                writer.add_scalars('Metrics', 
                                {'train loss': running_loss / update_step}, 
                                (epoch-1) * len(train_loader) + batch_idx)
#                 print("train loss for {} batches:".format(update_step), running_loss / update_step)

#                 # ...log a Matplotlib Figure showing the model's predictions on a
#                 # random mini-batch
#                 writer.add_figure('predictions vs. actuals',
#                                 plot_classes_preds(model, images, labels),
#                                 global_step=epoch * len(train_loader) + batch_idx)
                running_loss = 0.0
        
            
            
        else:
            with torch.no_grad():
                model.eval()
                list_outputs = []
                list_labels = []
                for images, labels in val_loader:
                    images, labels = Variable(images.to(device)), Variable(labels.to(device))
                    labels = labels.view(-1,1)

                    output = model(images)
                    loss = criterion(output, labels, cat='valid')
                    valid_loss += loss.item()*images.size(0)
                    
#                     scheduler.step(loss)
                    
                    # save output and labels for kappa_score
                    outputs = np.where(output.cpu().numpy() >= 0.5, 1, 0)
#                     outputs = np.squeeze(outputs)
                    true_labels = labels.cpu().numpy()#np.squeeze(labels.cpu().numpy())
                    list_outputs.extend(outputs.tolist())
                    list_labels.extend(true_labels.tolist())
<<<<<<< HEAD
=======
            print("validation list_outputs[:10]:", output.cpu().numpy()[:10], " labels: ", list_labels[:10])
>>>>>>> first-stage
                    
        # shuffle train loader
        train_loader.dataset.shuffle_dataset()
        
        # Calculate losses
        train_loss = train_loss/len(train_loader.sampler)
        train_losses.append(train_loss)
        valid_loss = valid_loss/len(val_loader.sampler)
        valid_losses.append(valid_loss)
        # ---- metrics
        # Kappa statistics
        kappa_score = cohen_kappa_score(list_outputs, list_labels)
        kappa_scores.append(kappa_score)
        # AUC ROC
        try:
            auc_roc_score = roc_auc_score(list_outputs, list_labels)
        except ValueError:
            auc_roc_score = 0
            
        auc_roc_scores.append(auc_roc_score)
        # accuracy
        accuracy = accuracy_score(list_outputs, list_labels)
        accuracy_scores.append(accuracy)
        
        print (f"\nTraining Loss : {train_loss} \nValidation Loss : {valid_loss} \nKappa score : {kappa_score} \nAUC ROC score : {auc_roc_score} \nAccuracy : {accuracy}")
        writer.add_scalars('Metrics', 
                                {'val loss': valid_loss,
                                'kappa score': kappa_score}, 
                                (epoch-1) * len(train_loader) + batch_idx)

        if valid_loss < valid_loss_min:
            print (f"Validation Loss decreased from {valid_loss_min} to  {valid_loss} ....Saving model")
            valid_loss_min = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
<<<<<<< HEAD
        torch.save(model.state_dict(), 'models/model_{}px_epoch-{}_kappa-{}.pt'.format(str(INPUT_SIZE), str(epoch+3), str(round(kappa_score, 3))))
=======
        torch.save(model.state_dict(), 'models/model_{}px_epoch-{}_kappa-{}.pt'.format(str(INPUT_SIZE), str(epoch), str(round(kappa_score, 3))))
>>>>>>> first-stage
        
            
        # free up memory
        del list_outputs
        del list_labels
        gc.collect()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
    plotify(train_losses, valid_losses, kappa_scores, auc_roc_scores)

    return model