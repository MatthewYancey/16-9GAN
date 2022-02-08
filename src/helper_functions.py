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


# helper function for converting images to a normal range
def apply_scale(img_tensor, plot=False):
    min_value = img_tensor.min()
    span = img_tensor.max() - img_tensor.min()
    img_tensor = (img_tensor - min_value) / span

    # tensorboard and matplotlib take images in different formats
    if plot:
        img_tensor = img_tensor.transpose(0, 2)
        img_tensor = torch.rot90(img_tensor, -1)  # fixes a rotation issue

    return img_tensor


def checkpoint(batch_counter,
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
               dataloader_train,
               train_reference_index,
               dataloader_val,
               val_reference_index,
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
    writer.add_scalar('Gen_Train/L2', gen_train_loss_l2, batch_counter)
    writer.add_scalar('Gen_val/Loss', gen_val_loss, batch_counter)
    writer.add_scalar('Gen_val/L2', gen_val_loss_l2, batch_counter)

    # saves a checkpoint
    checkpoint = {'gen_state': gen.state_dict(),
                  'gen_optimizer': optimizer_gen.state_dict(),
                  'disc_state': disc.state_dict(),
                  'disc_optimizer': optimizer_disc.state_dict(),
                  'batch_counter': batch_counter}
    torch.save(checkpoint, log_dir + 'checkpoint.pt')

    # gets images from the dataloader
    train_image = dataloader_train.dataset.__getitem__(train_reference_index)
    train_image = train_image.unsqueeze(0)
    train_image = train_image.cuda()
    train_image = apply_scale(train_image)
    val_image = dataloader_val.dataset.__getitem__(val_reference_index)
    val_image = val_image.unsqueeze(0)
    val_image = val_image.cuda()
    val_image = apply_scale(val_image)

    # if this is the first epoch we save the reference images
    if batch_counter == 0:
        print('Saving reference images')
        # training reference image
        writer.add_image('.Reference Train Image', train_image.squeeze(0))
        writer.add_image('.Reference Train Image Mask', apply_mask(train_image.squeeze(0), img_width, single_side))
        # validation reference image
        writer.add_image('.Reference Validation Image', val_image.squeeze(0))
        writer.add_image('.Reference Validation Image Mask', apply_mask(val_image.squeeze(0), img_width, single_side))

    # saves the progress of imagse
    with torch.no_grad():
        train_image_gen = gen(apply_mask(train_image, img_width, single_side))
        val_image_gen = gen(apply_mask(val_image, img_width, single_side))

    train_image_gen = train_image_gen.squeeze(0)
    train_image_gen = apply_scale(train_image_gen)
    train_image_gen = apply_comp(train_image.squeeze(0), train_image_gen, img_width, single_side)
    writer.add_image('train', train_image_gen, batch_counter)

    val_image_gen = val_image_gen.squeeze(0)
    val_image_gen = apply_scale(val_image_gen)
    val_image_gen = apply_comp(val_image.squeeze(0), val_image_gen, img_width, single_side)
    writer.add_image('val', val_image_gen, batch_counter)

    writer.close()
    print('Saved checkpoint')


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


def gpu_memory():
    print(f'Allocated memory: {torch.cuda.memory_allocated() / 10**9}')

