# Fin-GAN
Code to accompany the paper "Fin-GAN: Forecasting and Classifying Financial Time Series via Generative Adversarial Networks"

The Fin-GAN code and some other supplements are available here. The main code is in the Fin-GAN-online.py file, but you can also find the data cleaning file and the list of stocks and the corresponding ETFs. The data can be download from CRSP on WRDS. I saved all files as TICKER-data.csv, which is the notation used in the code. Please adjust to your own needs.

For Xavier initialisation the following lines can be uncommented:

        # nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        # nn.init.xavier_normal_(self.lstm.weight_hh_l0)

As mentioned in the paper, this makes the particular architecture less prone to mode collapse (in the BCE loss case specifically) and more stable, but you can also use He initialisation, which is the default initialisation in Pytorch. 

The notation in the code is the same as in the paper (Algorithm 1). Some variables might be unnecessary, eg epochs_val, just set it to True/False, if needed. This was left from the previous versions of the code, in order to make fewer changes. I tried to make comments here and there, but you will see that it is a long code. I hope you find it useful regardless.

Also, the results should be different depending on the stock used and the periods used for training, validation, and testing. Of course, they will also depend on the seed, but they should be consistent. Let me know if you have any questions by emailing at vuletic [at] maths [dot] ox [dot] ac [dot] uk.
