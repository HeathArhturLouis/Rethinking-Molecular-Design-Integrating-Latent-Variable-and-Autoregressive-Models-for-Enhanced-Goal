def epoch_train_vanilla(phase, epoch, ae, sample_idxes, data_binary, data_masks, data_property, cmd_args, optimizer_encoder=None, optimizer_decoder=None):
    '''
    No regularizer/regressor, just vanilla VAE

    Returns:
        - eq <-- the autoencoder model
        - avg_loss <-- average loss over all batches in the epoch
            [0] - Is the average vae_loss
            [1] - Is the average DKL loss
    '''

    total_vae_loss = []  # perp loss total per batch
    total_kl_loss = []  # KL loss total per batch

    pbar = tqdm(range(0, (len(sample_idxes) + (cmd_args.batch_size - 1)) // cmd_args.batch_size), unit='batch')

    if phase == 'train' and optimizer_encoder is not None:
        ae.train()
    else:
        ae.eval()

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * cmd_args.batch_size: (pos + 1) * cmd_args.batch_size]
        x_inputs, y_inputs, v_tb, v_ms, t_y = get_batch_input_vae(selected_idx, data_binary, data_masks, data_property)
        loss_list = ae.forward(x_inputs, y_inputs, v_tb, v_ms, t_y)
        perp_loss = loss_list[0]
        kl_loss = loss_list[1]

        total_loss_batch = perp_loss + cmd_args.kl_coeff * kl_loss  # KL Divergence weighting coeff (\beta?)

        pbar.set_description(f'Epoch: {epoch} Phase: {phase} - VAE Loss: {total_loss_batch.item():.5f} | Perp Loss {perp_loss.item()} | KDL Loss {kl_loss.item()} |')

        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad()
            if optimizer_decoder is not None:
                optimizer_decoder.zero_grad()

            total_loss_batch.backward()

            optimizer_encoder.step()
            if optimizer_decoder is not None:
                optimizer_decoder.step()

        # Collect loss components
        total_vae_loss.append(perp_loss.item() * len(selected_idx))
        total_kl_loss.append(kl_loss.item() * len(selected_idx))
        n_samples += len(selected_idx)

    # Calculate the average loss of a batch
    avg_vae_loss = sum(total_vae_loss) / n_samples
    avg_kl_loss = sum(total_kl_loss) / n_samples 
    avg_loss = [avg_vae_loss, avg_kl_loss]

    return ae, avg_loss