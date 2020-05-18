import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def matplotlib_imshow(img, one_channel=False):
    plt.figure(figsize=(40,10))
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
        
def plot_random_images_tensorboard(writer, trainloader):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    image_plot = images[0].numpy()
    # swap dim for plot (BGR->RGB)
    image_plot = np.moveaxis(image_plot, 0, -1)
    print("image_plot:", image_plot.shape)

    plt.imshow(image_plot)
    plt.show()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    
    # show images
    matplotlib_imshow(img_grid)#, one_channel=True)
    # write to tensorboard
    writer.add_image('four_x-ray_images', img_grid)

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            labels[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig