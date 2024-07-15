batch = adv_center.repeat((1000, 1, 1, 1))
noise=self.noise_sampling(1000,self.pdf_args).float().reshape(batch.shape)
noisy_input = torch.clip(batch + noise, 0.0, 1.0)
logits=self.base_classifier(noisy_input)
predictions = (logits).argmax(1)
torch.sum(predictions!=y)
labels=(predictions != y).int()
labels.sum()
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
import matplotlib.pyplot as plt
import matplotlib
images=logits.view(1000, -1).cpu().numpy()
transformed_images = tsne.fit_transform(images)
labels=labels.cpu().numpy()

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}"
})
plt.rcParams["font.family"] = "serif"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(dpi=300,figsize=(5, 5))
scatter = plt.scatter(transformed_images[labels==1, 0], transformed_images[labels==1, 1], c='r', label=r'{\bf Adversarial}', alpha=0.5)
scatter = plt.scatter(transformed_images[labels==0, 0], transformed_images[labels==0, 1], c='black', label=r'{\bf Benign}', alpha=0.5)
# plt.colorbar(scatter)
plt.title(r"{\bf $\mathbf{\underline{p_{adv}}=92.6\%}$, $\mathbf{p=90\%}$, empirical ASP: $\mathbf{94.2\%}$}",fontsize=15)
plt.xlabel('dimension 1')
plt.ylabel('dimension 2')
plt.legend(fontsize=15)
# plt.tight_layout()
plt.savefig('./paper_utils/p090_tsne.png', dpi=1200)