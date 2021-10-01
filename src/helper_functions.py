import torch
import shutil
from torch.utils.tensorboard import SummaryWriter


# helper function for apply the mask for cutting the frame to 4:3
def apply_mask(img_batch, img_width, single_side):
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
               train_loss,
               val_loss,
               log_dir,
               gen,
               optimizer_gen,
               disc,
               optimizer_disc,
               dataloader_train,
               train_reference_index,
               dataloader_val,
               val_reference_index,
               dataloader_test,
               test_reference_index,
               img_height,
               img_width,
               single_side):

    # saves loss to the tensorboard log
    writer = SummaryWriter(log_dir)
    writer.add_scalar('Loss/Disc', disc_loss, batch_counter)
    writer.add_scalar('Loss/Train', train_loss, batch_counter)
    writer.add_scalar('Loss/Val', val_loss, batch_counter)

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
    test_image = dataloader_test.dataset.__getitem__(test_reference_index)
    test_image = apply_padding(test_image, img_height, single_side)
    test_image = test_image.unsqueeze(0)
    test_image = test_image.cuda()
    test_image = apply_scale(test_image)

    # if this is the first epoch we save the reference images
    if batch_counter == 0:
        print('Saving reference images')
        # training reference image
        writer.add_image('.Reference Train Image', train_image.squeeze(0))
        writer.add_image('.Reference Train Image Mask', apply_mask(train_image.squeeze(0), img_width, single_side))
        # validation reference image
        writer.add_image('.Reference Validation Image', val_image.squeeze(0))
        writer.add_image('.Reference Validation Image Mask', apply_mask(val_image.squeeze(0), img_width, single_side))
        # testing reference image
        writer.add_image('.Reference Test Image', test_image.squeeze(0))

    with torch.no_grad():
        _, train_image_gen = gen(apply_mask(train_image, img_width, single_side))
        _, val_image_gen = gen(apply_mask(val_image, img_width, single_side))
        _, test_image_gen = gen(test_image)

    train_image_gen = train_image_gen.squeeze(0)
    train_image_gen = apply_scale(train_image_gen)
    train_image_gen = apply_comp(train_image.squeeze(0), train_image_gen, img_width, single_side)
    writer.add_image('train', train_image_gen, batch_counter)

    val_image_gen = val_image_gen.squeeze(0)
    val_image_gen = apply_scale(val_image_gen)
    val_image_gen = apply_comp(val_image.squeeze(0), val_image_gen, img_width, single_side)
    writer.add_image('val', val_image_gen, batch_counter)

    test_image_gen = test_image_gen.squeeze(0)
    test_image_gen = apply_scale(test_image_gen)
    test_image_gen = apply_comp(test_image.squeeze(0), test_image_gen, img_width, single_side)
    writer.add_image('test', test_image_gen, batch_counter)

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
