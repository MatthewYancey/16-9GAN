import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
import subprocess


# helper function for apply the mask for cutting the frame to 4:3
def apply_mask(img_batch, img_width, single_side):
    # checks if this is a single image or batch
    if len(img_batch.shape) > 3:
        img_batch[:, :, :, (img_width - single_side):] = -1
        img_batch[:, :, :, :single_side] = -1
    else:
        img_batch[:, :, (img_width - single_side):] = -1
        img_batch[:, :, :single_side] = -1

    return img_batch

# applies a mask to the cetner of the image
def apply_mask_center(img_batch, x_pos, y_pos, square_size):
    # checks if this is a single image or batch
    if len(img_batch.shape) > 3:
        img_batch[:, :, y_pos:y_pos + square_size, x_pos:x_pos + square_size] = -1
    else:
        img_batch[:, y_pos:y_pos + square_size, x_pos:x_pos + square_size] = -1

    return img_batch


# adds -1 padding to the sides of images that are in 4:3 aspect ratio
def apply_padding(img, img_height, single_side):
    padding = torch.zeros([3, img_height, single_side])
    padding = padding.new_full((3, img_height, single_side), -1)
    img_cat = torch.cat((padding, img, padding), 2)

    return img_cat


def apply_comp(img, img_gen, img_width, single_side):
    # checks if this is a single image or batch
    if len(img.shape) > 3:
        comp_img = torch.cat((img_gen[:, :, :, :single_side],
                              img[:, :, :, single_side:(img_width - single_side)],
                              img_gen[:, :, :, (img_width - single_side):]), 3)
    else:
        comp_img = torch.cat((img_gen[:, :, :single_side],
                              img[:, :, single_side:(img_width - single_side)],
                              img_gen[:, :, (img_width - single_side):]), 2)

    return comp_img


def apply_comp_center(img, img_gen, x_pos, y_pos, square_size):
    # checks if this is a single image or batch
    if len(img.shape) > 3:
        left_side = img[:, :, :, :x_pos]
        middle = torch.cat((img[:, :, :y_pos, x_pos:x_pos + square_size],
                            img_gen[:, :, y_pos:y_pos + square_size, x_pos:x_pos + square_size],
                            img[:, :, y_pos + square_size:, x_pos:x_pos + square_size]), 2)
        
        right_side = img[:, :, :, x_pos + square_size:]
        comp_img = torch.cat((left_side,
                              middle,
                              right_side), 3)
    else:
        left_side = img[:, :, :x_pos]
        middle = torch.cat((img[:, :y_pos, x_pos:x_pos + square_size],
                            img_gen[:, y_pos:y_pos + square_size, x_pos:x_pos + square_size],
                            img[:, y_pos + square_size:, x_pos:x_pos + square_size]), 1)
        right_side = img[:, :, x_pos + square_size:]
        
        comp_img = torch.cat((left_side,
                              middle,
                              right_side), 2)

    return comp_img


# helper function for converting images to a normal range
def apply_scale(img_tensor, plot=False):
    # min_value = img_tensor.min()
    # span = img_tensor.max() - img_tensor.min()
    img_tensor = (img_tensor - (-1)) / 2

    # tensorboard and matplotlib take images in different formats
    if plot:
        img_tensor = img_tensor.transpose(0, 2)
        img_tensor = torch.rot90(img_tensor, -1)  # fixes a rotation issue

    return img_tensor


def checkpoint(i,
               batch_counter,
               disc_loss,
               disc_accuracy,
               gen_train_loss,
               gen_train_loss_l2,
               gen_val_loss,
               gen_val_loss_l2,
               log_dir,
               gen,
               optimizer_gen,
               disc,
               optimizer_disc,
               dataloader,
               reference_images,
               img_height,
               img_width,
               single_side):

    # opens the writer
    writer = SummaryWriter(log_dir)

    # saves the gradients of the generator
    for tag, value in gen.named_parameters():
        if value.grad is not None:
            writer.add_histogram('gen/' + tag, value.grad.cpu(), batch_counter)

    # saves the gradients of the discriminator
    for tag, value in disc.named_parameters():
        if value.grad is not None:
            writer.add_histogram('disc/' + tag, value.grad.cpu(), batch_counter)

    # saves loss to the tensorboard log
    writer.add_scalar('Disc/Loss', disc_loss, batch_counter)
    writer.add_scalar('Disc/Accuracy', disc_accuracy, batch_counter)

    writer.add_scalar('Gen_Train/Loss', gen_train_loss, batch_counter)
    writer.add_scalar('Gen_Train/MSE', gen_train_loss_l2, batch_counter)
    writer.add_scalar('Gen_val/Loss', gen_val_loss, batch_counter)
    writer.add_scalar('Gen_val/MSE', gen_val_loss_l2, batch_counter)

    # Saves the reference images
    for image_index in reference_images:
        image = dataloader.dataset.__getitem__(image_index)
        image = image.unsqueeze(0)
        image = image.cuda()
        # image = apply_scale(image)

        # if this is the first epoch we save the reference images
        if batch_counter == 0:
            print('Saving reference images')
            # training reference image
            writer.add_image(f'.Reference Image/{image_index}', apply_scale(image.squeeze(0)))
            writer.add_image(f'.Reference Image Mask/{image_index}', apply_mask(apply_scale(image.squeeze(0)), img_width, single_side))

        # saves the progress of imagse
        with torch.no_grad():
            image_gen = gen(apply_mask(image, img_width, single_side))

        image_gen = image_gen.squeeze(0)
        image_gen = apply_comp(image.squeeze(0), image_gen, img_width, single_side)
        image_gen = apply_scale(image_gen)
        writer.add_image(f'Validation_{image_index}', image_gen, batch_counter)

    # saves a checkpoint if we are at a new epoch
    if i == 0:
        print('Saving checkpoint at new epoch')
        checkpoint = {'gen_state': gen.state_dict(),
                    'gen_optimizer': optimizer_gen.state_dict(),
                    'disc_state': disc.state_dict(),
                    'disc_optimizer': optimizer_disc.state_dict(),
                    'batch_counter': batch_counter}
        torch.save(checkpoint, log_dir + 'checkpoint.pt')

    writer.close()
    print('Saved to tensorboard')


def load_checkpoint(prev_checkpoint, log_dir, gen, optimizer_gen, disc=None, optimizer_disc=None):
    if prev_checkpoint is not None:
        # loads the model weights
        checkpoint = torch.load(prev_checkpoint)
        gen.load_state_dict(checkpoint['gen_state'])
        optimizer_gen.load_state_dict(checkpoint['gen_optimizer'])
        if disc is not None:
            disc.load_state_dict(checkpoint['disc_state'])
            optimizer_disc.load_state_dict(checkpoint['disc_optimizer'])

        # loads the batch counter and incraments it because we at the last epoch we did
        batch_counter = checkpoint['batch_counter']
        batch_counter += 1

        print(f'Loaded checkpoint from {prev_checkpoint}')

    elif prev_checkpoint is None:
        # remove all previous logs
        try:
            shutil.rmtree(log_dir)
            print('Folders removed')
        except FileNotFoundError:
            print('No log folder found')

        batch_counter = 0

    return gen, optimizer_gen, disc, optimizer_disc, batch_counter


def load_checkpoint_inference(prev_checkpoint, gen):
    if prev_checkpoint is not None:
        # loads the model weights
        checkpoint = torch.load(prev_checkpoint)
        gen.load_state_dict(checkpoint['gen_state'])
        print(f'Loaded checkpoint from {prev_checkpoint}')

    return gen

def gpu_memory():
    print(f'Allocated memory: {(torch.cuda.memory_allocated() / 10**9):.4f}')


def checkpoint_center_mask(i,
               batch_counter,
               disc_loss,
               disc_accuracy,
               gen_train_loss,
               gen_train_loss_l2,
               gen_val_loss,
               gen_val_loss_l2,
               log_dir,
               gen,
               optimizer_gen,
               disc,
               optimizer_disc,
               dataloader,
               reference_images,
               x_pos,
               y_pos,
               square_size):

    # opens the writer
    writer = SummaryWriter(log_dir)

    # saves the gradients of the generator
    for tag, value in gen.named_parameters():
        if value.grad is not None:
            writer.add_histogram('gen/' + tag, value.grad.cpu(), batch_counter)

    # saves the gradients of the discriminator
    for tag, value in disc.named_parameters():
        if value.grad is not None:
            writer.add_histogram('disc/' + tag, value.grad.cpu(), batch_counter)

    # saves loss to the tensorboard log
    writer.add_scalar('Disc/Loss', disc_loss, batch_counter)
    writer.add_scalar('Disc/Accuracy', disc_accuracy, batch_counter)

    writer.add_scalar('Gen_Train/Loss', gen_train_loss, batch_counter)
    writer.add_scalar('Gen_Train/MSE', gen_train_loss_l2, batch_counter)
    writer.add_scalar('Gen_val/Loss', gen_val_loss, batch_counter)
    writer.add_scalar('Gen_val/MSE', gen_val_loss_l2, batch_counter)

    # Saves the reference images
    for image_index in reference_images:
        image = dataloader.dataset.__getitem__(image_index)
        image = image.unsqueeze(0)
        image = image.cuda()
        # image = apply_scale(image)

        # if this is the first epoch we save the reference images
        if batch_counter == 0:
            print('Saving reference images')
            # training reference image
            writer.add_image(f'.Reference Image/{image_index}', apply_scale(image).squeeze(0))
            writer.add_image(f'.Reference Image Mask/{image_index}', apply_mask_center(apply_scale(image).squeeze(0), x_pos, y_pos, square_size))

        # saves the progress of imagse
        with torch.no_grad():
            image_gen = gen(apply_mask_center(image, x_pos, y_pos, square_size))

        image_gen = image_gen.squeeze(0)
        image_gen = apply_comp_center(image.squeeze(0), image_gen, x_pos, y_pos, square_size)
        image_gen = apply_scale(image_gen)
        writer.add_image(f'Validation_{image_index}', image_gen, batch_counter)

    # saves a checkpoint if we are at a new epoch
    if i == 0:
        print('Saving checkpoint at new epoch')
        checkpoint = {'gen_state': gen.state_dict(),
                    'gen_optimizer': optimizer_gen.state_dict(),
                    'disc_state': disc.state_dict(),
                    'disc_optimizer': optimizer_disc.state_dict(),
                    'batch_counter': batch_counter}
        torch.save(checkpoint, log_dir + 'checkpoint.pt')

    writer.close()
    print('Saved to tensorboard')

