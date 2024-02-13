import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time, os
from utils.augmentation.udagan import *
from utils.augmentation.dataloader import get_data
from utils.augmentation.aug_utils import *
# from module.model_sprites import *

eps = 1e-8

def train_udagan(parameters, device):

    dataloader, _ = get_data(batch_size=parameters['batch_size'], file=parameters['dataset_file'], ref_file=parameters['ref_data_file'])


    parameters['n_features'] = dataloader.dataset.tensors[0].size(-1)
    netG = Generator(latent_dim=parameters['num_z'], n_zim=parameters['n_zim'], input_dim=parameters['n_features']).to(device)
    netD = Discriminator(input_dim=parameters['n_features']).to(device)

    iter_num = len(dataloader)

    if parameters['initial_w']:
        print('use initial weigths')
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Loss functions
    criterionD = nn.BCELoss()
    mseDist = nn.MSELoss()

    # Set Adam optimiser for discriminator and augmenter
    optimD = optim.Adam([{'params': netD.parameters()}], lr=parameters['learning_rate'])
    optimG = optim.Adam([{'params': netG.parameters()}], lr=parameters['learning_rate'])

    real_label = 1.
    fake_label = 0.
    G_losses = []
    D_losses = []

    print('-'*50)
    print('Starting the training ...')

    for epoch in range(parameters['num_epochs']):
        epoch_start_time = time.time()
        G_loss_e, D_loss_e = 0, 0
        gen_loss_e, recon_loss_e = 0, 0
        n_adv = 0
        for i, (data, data_bin, ref_data, ref_data_bin) in enumerate(dataloader, 0):
            b_size = parameters['batch_size']
            data = data.to(device)
            data_bin = data_bin.to(device)
            ref_data = ref_data.to(device)
            ref_data_bin = ref_data_bin.to(device)
            # Updating the discriminator -----------------------------------
            optimD.zero_grad()
            # Original samples
            label = torch.full((b_size,), real_label, device=device)
            _, probs_real = netD(ref_data_bin)
            loss_real = criterionD(probs_real.view(-1), label)

            if F.relu(loss_real - np.log(2) / 2) > 0:
                loss_real.backward()
                optim_D = True
            else:
                optim_D = False

            # Generated samples
            label.fill_(fake_label)
            _, fake_data = netG(data, device)
            # binarizing the augmented sample
            if parameters['n_zim'] > 1:
                p_bern = ref_data_bin * fake_data[:, parameters['n_features']:]
                # fake_data_bin = torch.bernoulli(p_bern)
                fake_data = fake_data[:, :parameters['n_features']] * ref_data_bin
                fake_data_bin = 0. * fake_data
                fake_data_bin[fake_data > 1e-3] = 1.
            else:
                fake_data_bin = 0. * fake_data
                fake_data_bin[fake_data > 1e-3] = 1.

            _, probs_fake = netD(fake_data_bin.detach())
            loss_fake = criterionD(probs_fake.view(-1), label)

            if F.relu(loss_fake - np.log(2) / 2) > 0:
                loss_fake.backward()
                optim_D = True

            # Loss value for the discriminator
            D_loss = loss_real + loss_fake

            if optim_D:
                optimD.step()
            else:
                n_adv += 1

            # Updating the augmenter ---------------------------------------
            optimG.zero_grad()
            # Augmented data treated as real data
            z, probs_fake = netD(fake_data_bin)
            label.fill_(real_label)
            gen_loss = criterionD(probs_fake.view(-1), label)
            recon_loss = (F.mse_loss(fake_data, data, reduction='mean') + criterionD(fake_data_bin, data_bin)) / 2

            # Loss value for the augmenter
            G_loss = parameters['lambda'][0] * gen_loss + parameters['lambda'][1] * recon_loss
            G_loss.backward()
            optimG.step()

            G_losses.append(G_loss.data.item())
            D_losses.append(D_loss.data.item())
            G_loss_e += G_loss.data.item()
            D_loss_e += D_loss.data.item()
            gen_loss_e += gen_loss.data.item()
            recon_loss_e += recon_loss.data.item()

        G_loss_epoch = G_loss_e / (iter_num)
        D_loss_epoch = D_loss_e / (iter_num )
        gen_loss_epoch = gen_loss_e / (iter_num)
        recon_loss_epoch = recon_loss_e / (iter_num)

        print('=====> Epoch:{}, Generator Loss: {:.4f}, Discriminator Loss: {'':.4f}, Recon Loss: {:.4f}, Elapsed Time:{:.2f}'.format(epoch, G_loss_epoch, D_loss_epoch, recon_loss_epoch, time.time() - epoch_start_time))

    print("-" * 50)
    # Save trained models
    if parameters['save']:

        torch.save({
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
            'parameters': parameters
            }, parameters['file_name'])

        # Plot the training losses.
        plt.figure()
        plt.title("Augmenter and Discriminator Loss Values in Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(parameters['saving_path'] + 'loss_curve.png')
