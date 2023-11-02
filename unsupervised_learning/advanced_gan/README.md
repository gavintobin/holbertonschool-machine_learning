data

https://wandb.ai/gavint/your_project_name/runs/el36nykh?workspace=user-gavintobin714



advanced dcgan project using Fashion mnist dataset built with pytorch mainly


baseline model perameters

lr=0.0001
epochs=50
loss_function =nn.BCELoss()

optim_gen= torch.optim.Adam(generator.parameters(), lr=lr)
optim_dis= torch.optim.Adam(discriminate.parameters(), lr=lr)

BASELINE

for baseline dc gan optimizer chosen was adam. for simplicity purposes, little memory requirements and computationally efficient. it also was chosen bc it works betterwith nosier gradients.This baseline model was one of the more efficent models, the images looked the best in this expirement

ARCHVAR

in this experiment i started with changing the number of layers by adding one more than two more. This made the model go crazy.  it started off okay but not neccisarily good then around 10 epochs it went up to the 20s and stayed around that range until 20 epooch range then jumped and stayed to around the 90s which is way off since it needs to be as lose to 1 as possible. the dis wasnt as bad stayed around the 10s range consistantly

HYPERPERAM


the epochs were changed to 64 from 50 the learning rate decreased by a decimal point and optimizer was switched to sgd form adam.
L and G behaved similarly to baseline got better over time and then started to do worse the longer it went on.


ADVANCED BASELINE


best performing model so far. similiar to dc gan baseline. within 10 epocs gen was in 5s which isnt necisarrily good. seemed tod dial in after that with loss staying belove .5 and gen staying in low 1s


ADVANCED  HYPERPERAMS
tried to change layers again and just like reg dcgan it completely threw my model off. i belive this is because ive added more trainable perameters to my model by adding more layers and i didnt train it enough. i also tried lowering the learning rate but it was not of muh help. made it a little better but nothing wortht it.


ADVANVED  ARCHVAR
this time i tried using less layers and it made my model worse than baseline. the adding and decreasing of the layers were the things that gave me the most issues and made my model act the craziest


TRANSFER OF KNOWLEGDE
  i learned from the first proj that the baseline worked the best for me so i tried to see if just changing the dataset woudl work and it did. suprisingly the advanced baseline worked well too. i had somewhat similar findings or outcomes when working in both projects. in the the adncaed i tried to do the opposite of the tweaks i made in the first proj just to see what would happen. overall i learned that what i started with was the best model scenario so far




