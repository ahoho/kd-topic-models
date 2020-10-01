import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.init import kaiming_uniform_, xavier_uniform_


class Scholar(object):
    def __init__(
        self,
        config,
        alpha=1.0,
        learning_rate=0.001,
        init_embeddings=None,
        init_bg=None,
        update_background=True,
        adam_beta1=0.99,
        adam_beta2=0.999,
        device=None,
        seed=None,
        classify_from_covars=True,
        classify_from_topics=True,
        classify_from_doc_reps=True,
    ):

        """
        Create the model
        :param config: a dictionary with the model configuration
        :param alpha: hyperparameter for the document representation prior
        :param learning_rate: learning rate for Adam
        :param init_embeddings: a matrix of embeddings to initialize the first layer of the bag-of-words encoder
        :param update_embeddings: if True, update word embeddings during training
        :param init_bg: a vector of empirical log backgound frequencies
        :param update_background: if True, update the background term during training
        :param adam_beta1: first hyperparameter for Adam
        :param adam_beta2: second hyperparameter for Adam
        :param device: (int) the number of the GPU to use
        """
        self._call = locals()
        self._call.pop('self')
        self._call.pop('init_embeddings')

        if seed is not None:
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        self.network_architecture = config
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1

        self.update_background = update_background

        # create priors on the hidden state
        self.n_topics = config["n_topics"]

        if device is None:
            self.device = "cpu"
        else:
            self.device = "cuda:" + str(device)

        # interpret alpha as either a (symmetric) scalar prior or a vector prior
        if np.array(alpha).size == 1:
            # if alpha is a scalar, create a symmetric prior vector
            self.alpha = alpha * np.ones((1, self.n_topics)).astype(np.float32)
        else:
            # otherwise use the prior as given
            self.alpha = np.array(alpha).astype(np.float32)
            assert len(self.alpha) == self.n_topics

        # create the pyTorch model
        self._model = torchScholar(
            config,
            self.alpha,
            init_emb=init_embeddings,
            bg_init=init_bg,
            device=self.device,
            classify_from_covars=classify_from_covars,
            classify_from_topics=classify_from_topics,
            classify_from_doc_reps=classify_from_doc_reps,
        ).to(self.device)

        # set the criterion
        self.criterion = nn.BCEWithLogitsLoss()

        # create the optimizer
        grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
        self.optimizer = optim.Adam(
            grad_params, lr=learning_rate, betas=(adam_beta1, adam_beta2)
        )

    def fit(
        self,
        X,
        Y,
        PC,
        TC,
        DR,
        eta_bn_prop=1.0,
        l1_beta=None,
        l1_beta_c=None,
        l1_beta_ci=None,
    ):
        """
        Fit the model to a minibatch of data
        :param X: np.array of document word counts [batch size x vocab size]
        :param Y: np.array of labels [batch size x n_labels]
        :param PC: np.array of prior covariates influencing the document-topic prior [batch size x n_prior_covars]
        :param TC: np.array of topic covariates to be associated with topical deviations [batch size x n_topic_covars]
        :param DR: np.array of document representations [batch size x doc_dim]
        :param l1_beta: np.array of prior variances on the topic weights
        :param l1_beta_c: np.array of prior variances on the weights for topic covariates
        :param l1_beta_ci: np.array of prior variances on the weights for topic-covariate interactions
        :return: loss; label pred probs; document representations; neg-log-likelihood; KLD
        """
        # move data to device
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if DR is not None:
            DR = torch.Tensor(DR).to(self.device)
        self.optimizer.zero_grad()

        # do a forward pass
        thetas, X_recon, Y_probs, losses = self._model(
            X,
            Y,
            PC,
            TC,
            DR,
            eta_bn_prop=eta_bn_prop,
            l1_beta=l1_beta,
            l1_beta_c=l1_beta_c,
            l1_beta_ci=l1_beta_ci,
        )
        loss, nl, kld = losses
        # update model
        loss.backward()
        self.optimizer.step()

        if Y_probs is not None:
            Y_probs = Y_probs.to("cpu").detach().numpy()
        return (
            loss.to("cpu").detach().numpy(),
            Y_probs,
            thetas.to("cpu").detach().numpy(),
            nl.to("cpu").detach().numpy(),
            kld.to("cpu").detach().numpy(),
        )

    def predict(self, X, PC, TC, DR, eta_bn_prop=0.0):
        """
        Predict labels for a minibatch of data
        """
        # input a vector of all zeros in place of the labels that the model has been trained on
        batch_size = self.get_batch_size(X)
        Y = np.zeros((batch_size, self.network_architecture["n_labels"])).astype(
            "float32"
        )
        X = torch.Tensor(X).to(self.device)
        Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if DR is not None:
            DR = torch.Tensor(DR).to(self.device)
        theta, _, Y_recon, _ = self._model(
            X, Y, PC, TC, DR, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop
        )
        return theta, Y_recon.to("cpu").detach().numpy()

    def predict_from_topics(self, theta, PC, TC, DR, eta_bn_prop=0.0):
        """
        Predict label probabilities from each topic
        """
        theta = torch.Tensor(theta)
        if PC is not None:
            PC = torch.Tensor(PC)
        if TC is not None:
            TC = torch.Tensor(TC)
        if DR is not None:
            DR = torch.Tensor(DR)
        probs = self._model.predict_from_theta(theta, PC, TC, DR)
        return probs.to("cpu").detach().numpy()

    def get_losses(self, X, Y, PC, TC, DR, eta_bn_prop=0.0, n_samples=0):
        """
        Compute and return the loss values for all instances in X, Y, PC, and TC averaged over multiple samples
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        if DR is not None and batch_size == 1:
            DR = np.expand_dims(DR, axis=0)
        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if DR is not None:
            DR = torch.Tensor(DR).to(self.device)
        if n_samples == 0:
            _, _, _, temp = self._model(
                X, Y, PC, TC, DR, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop
            )
            loss, NL, KLD = temp
            losses = loss.to("cpu").detach().numpy()
        else:
            _, _, _, temp = self._model(
                X, Y, PC, TC, DR, do_average=False, var_scale=1.0, eta_bn_prop=eta_bn_prop
            )
            loss, NL, KLD = temp
            losses = loss.to("cpu").detach().numpy()
            for s in range(1, n_samples):
                _, _, _, temp = self._model(
                    X,
                    Y,
                    PC,
                    TC,
                    DR,
                    do_average=False,
                    var_scale=1.0,
                    eta_bn_prop=eta_bn_prop,
                )
                loss, NL, KLD = temp
                losses += loss.to("cpu").detach().numpy()
            losses /= np.float32(n_samples)

        return losses

    def compute_theta(self, X, Y, PC, TC, DR, eta_bn_prop=0.0):
        """
        Return the latent document representation (mean of posterior of theta) for a given batch of X, Y, PC, and TC
        """
        batch_size = self.get_batch_size(X)
        if batch_size == 1:
            X = np.expand_dims(X, axis=0)
        if Y is not None and batch_size == 1:
            Y = np.expand_dims(Y, axis=0)
        if PC is not None and batch_size == 1:
            PC = np.expand_dims(PC, axis=0)
        if TC is not None and batch_size == 1:
            TC = np.expand_dims(TC, axis=0)
        if DR is not None and batch_size == 1:
            DR = np.expand_dims(DR, axis=0)

        X = torch.Tensor(X).to(self.device)
        if Y is not None:
            Y = torch.Tensor(Y).to(self.device)
        if PC is not None:
            PC = torch.Tensor(PC).to(self.device)
        if TC is not None:
            TC = torch.Tensor(TC).to(self.device)
        if DR is not None:
            DR = torch.Tensor(DR).to(self.device)
        theta, _, _, _ = self._model(
            X, Y, PC, TC, DR, do_average=False, var_scale=0.0, eta_bn_prop=eta_bn_prop
        )

        return theta.to("cpu").detach().numpy()

    def get_weights(self):
        """
        Return the topic-vocabulary deviation weights
        """
        emb = self._model.beta_layer.to("cpu").weight.detach().numpy().T
        self._model.beta_layer.to(self.device)
        return emb

    def get_bg(self):
        """
        Return the background terms
        """
        bg = self._model.beta_layer.to("cpu").bias.detach().numpy()
        self._model.beta_layer.to(self.device)
        return bg

    def get_prior_weights(self):
        """
        Return the weights associated with the prior covariates
        """
        emb = self._model.prior_covar_weights.to("cpu").weight.detach().numpy().T
        self._model.prior_covar_weights.to(self.device)
        return emb

    def get_covar_weights(self):
        """
        Return the topic weight (deviations) associated with the topic covariates
        """
        emb = self._model.beta_c_layer.to("cpu").weight.detach().numpy().T
        self._model.beta_c_layer.to(self.device)
        return emb

    def get_covar_interaction_weights(self):
        """
        Return the weights (deviations) associated with the topic-covariate interactions
        """
        emb = self._model.beta_ci_layer.to("cpu").weight.detach().numpy().T
        self._model.beta_ci_layer.to(self.device)
        return emb

    def get_batch_size(self, X):
        """
        Get the batch size for a minibatch of data
        :param X: the minibatch
        :return: the size of the minibatch
        """
        if len(X.shape) == 1:
            batch_size = 1
        else:
            batch_size, _ = X.shape
        return batch_size

    def eval(self):
        self._model.eval()

    def train(self):
        self._model.train()


class torchScholar(nn.Module):
    def __init__(
        self,
        config,
        alpha,
        init_emb=None,
        bg_init=None,
        device="cpu",
        classify_from_covars=True,
        classify_from_topics=True,
        classify_from_doc_reps=True,
    ):
        super(torchScholar, self).__init__()

        # load the configuration
        self.vocab_size = config["vocab_size"]
        self.words_emb_dim = config["embedding_dim"]
        self.zero_out_embeddings = config["zero_out_embeddings"]        
        self.reconstruct_bow = config["reconstruct_bow"]
        self.doc_reps_dim = config["doc_reps_dim"]
        self.attend_over_doc_reps = config["attend_over_doc_reps"]
        self.use_doc_layer = config["use_doc_layer"]
        self.doc_reconstruction_weight = config["doc_reconstruction_weight"]
        self.doc_reconstruction_temp = config["doc_reconstruction_temp"]
        self.doc_reconstruction_min_count = config["doc_reconstruction_min_count"]
        self.n_topics = config["n_topics"]
        self.n_labels = config["n_labels"]
        self.n_prior_covars = config["n_prior_covars"]
        self.n_topic_covars = config["n_topic_covars"]
        self.classifier_layers = config["classifier_layers"]
        self.classifier_loss_weight = config["classifier_loss_weight"]
        self.use_interactions = config["use_interactions"]
        self.l1_beta_reg = config["l1_beta_reg"]
        self.l1_beta_c_reg = config["l1_beta_c_reg"]
        self.l1_beta_ci_reg = config["l1_beta_ci_reg"]
        self.l2_prior_reg = config["l2_prior_reg"]
        self.device = device
        self.classify_from_covars = classify_from_covars
        self.classify_from_topics = classify_from_topics
        self.classify_from_doc_reps = classify_from_doc_reps

        # create a layer for prior covariates to influence the document prior
        if self.n_prior_covars > 0:
            self.prior_covar_weights = nn.Linear(
                self.n_prior_covars, self.n_topics, bias=False
            )
        else:
            self.prior_covar_weights = None

        # create the encoder    
        emb_size = self.words_emb_dim
        classifier_input_dim = 0
        if self.classify_from_topics:
            classifier_input_dim = self.n_topics
        if self.n_prior_covars > 0:
            emb_size += self.n_prior_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_prior_covars
        if self.n_topic_covars > 0:
            emb_size += self.n_topic_covars
            if self.classify_from_covars:
                classifier_input_dim += self.n_topic_covars
        if self.doc_reps_dim is not None:
            if self.attend_over_doc_reps:
                self.attention_vec = torch.nn.Parameter(
                    torch.rand(self.doc_reps_dim)
                ).to(self.device)
            if self.use_doc_layer:
                emb_size += self.words_emb_dim
                self.doc_layer = nn.Linear(
                    self.doc_reps_dim, self.words_emb_dim
                ).to(self.device)
            else:
                emb_size += self.doc_reps_dim
            if self.classify_from_doc_reps:
                classifier_input_dim += self.doc_reps_dim
        if self.n_labels > 0:
            emb_size += self.n_labels

        self.encoder_dropout_layer = nn.Dropout(p=0.2)
        
        self.embeddings_x = torch.nn.ParameterDict()
        # initialize each embedding
        for emb_name, (emb_data, update) in init_emb.items():
            self.embeddings_x[emb_name] = torch.nn.Parameter(
                torch.zeros(
                    size=(self.words_emb_dim, self.vocab_size)
                ).to(self.device),
                requires_grad=update,
            )
            if emb_data is not None:
                (self.embeddings_x[emb_name]
                     .data.copy_(torch.from_numpy(emb_data)).to(self.device)
                )
            else:
                kaiming_uniform_(self.embeddings_x[emb_name], a=np.sqrt(5))         
                xavier_uniform_(self.embeddings_x[emb_name])
        
        # create the mean and variance components of the VAE
        self.mean_layer = nn.Linear(emb_size, self.n_topics)
        self.logvar_layer = nn.Linear(emb_size, self.n_topics)

        self.mean_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.mean_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))
        ).to(self.device)
        self.mean_bn_layer.weight.requires_grad = False
        self.logvar_bn_layer = nn.BatchNorm1d(
            self.n_topics, eps=0.001, momentum=0.001, affine=True
        )
        self.logvar_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.n_topics))
        ).to(self.device)
        self.logvar_bn_layer.weight.requires_grad = False

        self.z_dropout_layer = nn.Dropout(p=0.2)

        # create the decoder
        self.beta_layer = nn.Linear(self.n_topics, self.vocab_size)

        xavier_uniform_(self.beta_layer.weight)
        if bg_init is not None:
            self.beta_layer.bias.data.copy_(torch.from_numpy(bg_init))
            self.beta_layer.bias.requires_grad = False
        self.beta_layer = self.beta_layer.to(self.device)

        if self.n_topic_covars > 0:
            self.beta_c_layer = nn.Linear(
                self.n_topic_covars, self.vocab_size, bias=False
            ).to(self.device)
            if self.use_interactions:
                self.beta_ci_layer = nn.Linear(
                    self.n_topics * self.n_topic_covars, self.vocab_size, bias=False
                ).to(self.device)

        # create the classifier
        if self.n_labels > 0:
            if self.classifier_layers == 0:
                self.classifier_layer_0 = nn.Linear(
                    classifier_input_dim, self.n_labels
                ).to(self.device)
            else:
                self.classifier_layer_0 = nn.Linear(
                    classifier_input_dim, classifier_input_dim
                ).to(self.device)
                self.classifier_layer_1 = nn.Linear(
                    classifier_input_dim, self.n_labels
                ).to(self.device)

        # create a final batchnorm layer
        self.eta_bn_layer = nn.BatchNorm1d(
            self.vocab_size, eps=0.001, momentum=0.001, affine=True
        ).to(self.device)
        self.eta_bn_layer.weight.data.copy_(
            torch.from_numpy(np.ones(self.vocab_size)).to(self.device)
        )
        self.eta_bn_layer.weight.requires_grad = False

        # create the document prior terms
        prior_mean = (np.log(alpha).T - np.mean(np.log(alpha), 1)).T
        prior_var = (
            ((1.0 / alpha) * (1 - (2.0 / self.n_topics))).T
            + (1.0 / (self.n_topics * self.n_topics)) * np.sum(1.0 / alpha, 1)
        ).T

        prior_mean = np.array(prior_mean).reshape((1, self.n_topics))
        prior_logvar = np.array(np.log(prior_var)).reshape((1, self.n_topics))
        self.prior_mean = torch.from_numpy(prior_mean).to(self.device)
        self.prior_mean.requires_grad = False
        self.prior_logvar = torch.from_numpy(prior_logvar).to(self.device)
        self.prior_logvar.requires_grad = False

    def forward(
        self,
        X,
        Y,
        PC,
        TC,
        DR,
        compute_loss=True,
        do_average=True,
        eta_bn_prop=1.0,
        var_scale=1.0,
        l1_beta=None,
        l1_beta_c=None,
        l1_beta_ci=None,
    ):
        """
        Do a forward pass of the model
        :param X: np.array of word counts [batch_size x vocab_size]
        :param Y: np.array of labels [batch_size x n_classes]
        :param PC: np.array of covariates influencing the prior [batch_size x n_prior_covars]
        :param TC: np.array of covariates with explicit topic deviations [batch_size x n_topic_covariates]
        :param DR: np.array of document representations [batch_size x doc_reps_dim]
        :param compute_loss: if True, compute and return the loss
        :param do_average: if True, average the loss over the minibatch
        :param eta_bn_prop: (float) a weight between 0 and 1 to interpolate between using and not using the final batchnorm layer
        :param var_scale: (float) a parameter which can be used to scale the variance of the random noise in the VAE
        :param l1_beta: np.array of prior variances for the topic weights
        :param l1_beta_c: np.array of prior variances on topic covariate deviations
        :param l1_beta_ci: np.array of prior variances on topic-covariate interactions
        :return: document representation; reconstruction; label probs; (loss, if requested)
        """
        en0_x = []

        deviation_covar_idx = 0
        for emb_name, embedding in self.embeddings_x.items():
            mapped_embeddings = torch.mm(X, embedding.T)
            if 'background' == emb_name:
                en0_x.append(mapped_embeddings)
            else:
                deviation_covar = PC[:, deviation_covar_idx].view(-1, 1)
                en0_x.append(mapped_embeddings * deviation_covar)
                deviation_covar_idx += 1
        en0_x = torch.stack(en0_x).mean(0)
        if self.zero_out_embeddings:
            en0_x = en0_x * 0
        encoder_parts = [en0_x]
        
        # append additional components to the encoder, if given
        if self.n_prior_covars > 0:
            encoder_parts.append(PC)
        if self.n_topic_covars > 0:
            encoder_parts.append(TC)
        if self.doc_reps_dim is not None:
            dr_out = torch.clamp(DR, min=-40)

            if self.attend_over_doc_reps:
                mask = dr_out[:, :, 0] == 0

                # do masked softmax                
                attn_weights = torch.matmul(dr_out, self.attention_vec)
                attn = torch.softmax(attn_weights.masked_fill(mask, -1e32), dim=-1)
                dr_out = (dr_out * attn.unsqueeze(-1)).sum(1) # TODO: bmm instead?
                
            if self.use_doc_layer:
                dr_out = F.softplus(self.doc_layer(dr_out))
            
            encoder_parts.append(dr_out)
        if self.n_labels > 0:
            encoder_parts.append(Y)

        if len(encoder_parts) > 1:
            en0 = torch.cat(encoder_parts, dim=1).to(self.device)
        else:
            en0 = en0_x

        encoder_output = F.softplus(en0)
        encoder_output_do = self.encoder_dropout_layer(encoder_output)

        # compute the mean and variance of the document posteriors
        posterior_mean = self.mean_layer(encoder_output_do)
        posterior_logvar = self.logvar_layer(encoder_output_do)

        posterior_mean_bn = self.mean_bn_layer(posterior_mean)
        posterior_logvar_bn = self.logvar_bn_layer(posterior_logvar)
        # posterior_mean_bn = posterior_mean
        # posterior_logvar_bn = posterior_logvar

        posterior_var = posterior_logvar_bn.exp().to(self.device)

        # sample noise from a standard normal
        eps = X.data.new().resize_as_(posterior_mean_bn.data).normal_().to(self.device)

        # compute the sampled latent representation
        z = posterior_mean_bn + posterior_var.sqrt() * eps * var_scale
        z_do = self.z_dropout_layer(z)

        # pass the document representations through a softmax
        theta = F.softmax(z_do, dim=1)

        # combine latent representation with topics and background
        # beta layer here includes both the topic weights and the background term (as a bias)
        eta = self.beta_layer(theta)

        # add deviations for covariates (and interactions)
        if self.n_topic_covars > 0:
            eta = eta + self.beta_c_layer(TC)
            if self.use_interactions:
                theta_rsh = theta.unsqueeze(2)
                tc_emb_rsh = TC.unsqueeze(1)
                covar_interactions = theta_rsh * tc_emb_rsh
                batch_size, _, _ = covar_interactions.shape
                eta += self.beta_ci_layer(
                    covar_interactions.reshape(
                        (batch_size, self.n_topics * self.n_topic_covars)
                    )
                )

        # pass the unnormalized word probabilities through a batch norm layer
        eta_bn = self.eta_bn_layer(eta)
        # eta_bn = eta

        # compute X recon with and without batchnorm on eta, and take a convex combination of them
        X_recon_bn = F.softmax(eta_bn, dim=1)
        X_recon_no_bn = F.softmax(eta, dim=1)
        X_recon = eta_bn_prop * X_recon_bn + (1.0 - eta_bn_prop) * X_recon_no_bn

        # reconstruct the document representation
        X_soft_recon = None
        if self.doc_reconstruction_temp is not None:
            X_soft_recon = (
                eta_bn_prop * F.softmax(eta_bn / self.doc_reconstruction_temp, dim=1) 
                + (1.0 - eta_bn_prop) * F.softmax(eta / self.doc_reconstruction_temp, dim=1) 
            )
        
        # predict labels
        Y_recon = None
        if self.n_labels > 0:
            
            classifier_inputs = []
            if self.classify_from_topics:
                classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if self.classify_from_doc_reps:
                classifier_inputs.append(DR)

            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = classifier_inputs[0]

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_2(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        # compute the document prior if using prior covariates
        if self.n_prior_covars > 0:
            prior_mean = self.prior_covar_weights(PC)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        else:
            prior_mean = self.prior_mean.expand_as(posterior_mean)
            prior_logvar = self.prior_logvar.expand_as(posterior_logvar)

        if compute_loss:
            return (
                theta,
                X_recon,
                Y_recon,
                self._loss(
                    X,
                    Y,
                    DR,
                    X_recon,
                    Y_recon,
                    X_soft_recon,
                    prior_mean,
                    prior_logvar,
                    posterior_mean_bn,
                    posterior_logvar_bn,
                    do_average,
                    l1_beta,
                    l1_beta_c,
                    l1_beta_ci,
                ),
            )
        else:
            return theta, X_recon, Y_recon

    def _loss(
        self,
        X,
        Y,
        DR,
        X_recon,
        Y_recon,
        X_soft_recon,
        prior_mean,
        prior_logvar,
        posterior_mean,
        posterior_logvar,
        do_average=True,
        l1_beta=None,
        l1_beta_c=None,
        l1_beta_ci=None,
    ):

        NL = 0.
        # compute bag-of-words reconstruction loss
        if self.reconstruct_bow:
            NL += -(X * (X_recon + 1e-10).log()).sum(1)

        # knowledge distillation
        if X_soft_recon is not None:
            alpha = self.doc_reconstruction_weight
            t = self.doc_reconstruction_temp
            
            X_soft = torch.softmax(DR / t, dim=-1) * X.sum(1, keepdim=True) # multiply probabilities by counts
            X_soft = X_soft * (X_soft > self.doc_reconstruction_min_count).float()

            kd_loss = (alpha * t * t) * -(X_soft * (X_soft_recon + 1e-10).log()).sum(1)
            standard_loss = (1 - alpha) * -(X * (X_recon + 1e-10).log()).sum(1)

            # overwrite the NL loss (more of a safeguard than anything)
            NL = kd_loss + standard_loss
        
        # compute label loss
        if self.n_labels > 0:
            NL += -(Y * (Y_recon + 1e-10).log()).sum(1) * self.classifier_loss_weight
        
        # compute KLD
        prior_var = prior_logvar.exp()
        posterior_var = posterior_logvar.exp()
        var_division = posterior_var / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar

        # put KLD together
        KLD = 0.5 * (
            (var_division + diff_term + logvar_division).sum(1) - self.n_topics
        )

        # combine
        loss = NL + KLD

        # add regularization on prior
        if self.l2_prior_reg > 0 and self.n_prior_covars > 0:
            loss += (
                self.l2_prior_reg * torch.pow(self.prior_covar_weights.weight, 2).sum()
            )

        # add regularization on topic and topic covariate weights
        if self.l1_beta_reg > 0 and l1_beta is not None:
            l1_strengths_beta = torch.from_numpy(l1_beta).to(self.device)
            beta_weights_sq = torch.pow(self.beta_layer.weight, 2)
            loss += self.l1_beta_reg * (l1_strengths_beta * beta_weights_sq).sum()

        if self.n_topic_covars > 0 and l1_beta_c is not None and self.l1_beta_c_reg > 0:
            l1_strengths_beta_c = torch.from_numpy(l1_beta_c).to(self.device)
            beta_c_weights_sq = torch.pow(self.beta_c_layer.weight, 2)
            loss += self.l1_beta_c_reg * (l1_strengths_beta_c * beta_c_weights_sq).sum()

        if (
            self.n_topic_covars > 0
            and self.use_interactions
            and l1_beta_c is not None
            and self.l1_beta_ci_reg > 0
        ):
            l1_strengths_beta_ci = torch.from_numpy(l1_beta_ci).to(self.device)
            beta_ci_weights_sq = torch.pow(self.beta_ci_layer.weight, 2)
            loss += (
                self.l1_beta_ci_reg * (l1_strengths_beta_ci * beta_ci_weights_sq).sum()
            )

        # average losses if desired
        if do_average:
            return loss.mean(), NL.mean(), KLD.mean()
        else:
            return loss, NL, KLD

    def predict_from_theta(self, theta, PC, TC, DR):
        # Predict labels from a distribution over topics
        Y_recon = None
        if self.n_labels > 0:

            classifier_inputs = []
            if self.classify_from_topics:
                classifier_inputs = [theta]
            if self.classify_from_covars:
                if self.n_prior_covars > 0:
                    classifier_inputs.append(PC)
                if self.n_topic_covars > 0:
                    classifier_inputs.append(TC)
            if self.classify_from_doc_reps:
                classifier_inputs.append(DR)
            
            if len(classifier_inputs) > 1:
                classifier_input = torch.cat(classifier_inputs, dim=1).to(self.device)
            else:
                classifier_input = classifier_inputs[0].to(self.device)

            if self.classifier_layers == 0:
                decoded_y = self.classifier_layer_0(classifier_input)
            elif self.classifier_layers == 1:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                decoded_y = self.classifier_layer_1(cls0_sp)
            else:
                cls0 = self.classifier_layer_0(classifier_input)
                cls0_sp = F.softplus(cls0)
                cls1 = self.classifier_layer_1(cls0_sp)
                cls1_sp = F.softplus(cls1)
                decoded_y = self.classifier_layer_1(cls1_sp)
            Y_recon = F.softmax(decoded_y, dim=1)

        return Y_recon
