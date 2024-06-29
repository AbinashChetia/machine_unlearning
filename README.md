# Machine Unlearning
Machine unlearning is a process of removing the knowledge from a machine learning model. This is a very important process in the context of privacy and security. Here, neuron masking is implemented as an approach for machine unlearning, in case of a simple neural network model w.r.t. MNIST dataset and a VGG16 model w.r.t. CIFAR10 dataset.

This project is implemented as a part of the [ACM India Summer School 2024 on "Responsible and Safe AI"](https://precogatiiit.notion.site/ACM-India-Summer-School-on-Responsible-Safe-AI-76108c53564d4dc4af46c1d3bed52946), hosted by IIT Madras and sponsored by Centre for Responsible AI (CeRAI), IIT Madras, from 3rd June to 14th June 2024.

## Results
Confusion matrices generated for a trained VGG16 model on CIFAR10 dataset, before and after neuron masking (for the `plane` class), are shown below respectively:
![Before Neuron Masking](/results/cifar10_vgg16_cf.png)
![After Neuron Masking](/results/cifar10_unl_vgg16_cf.png)

## Dependencies
- Python 3.11
- Pytorch
- Torchvision
- Scikit-learn
- Numpy
- Pandas
- Matplotlib
- seaborn
- pickle
- tqdm

## Contributors
- [Abinash Chetia](https://github.com/AbinashChetia)
- [Anurag Lengure](https://github.com/anulengure5)
- [Anushka Nehra](https://github.com/Nehra-cell)
- [Bettina Ninan](https://github.com/Bettina2004)
- [Tehseen Shaikh](https://github.com/Tehseen-dataenthusiastic)
