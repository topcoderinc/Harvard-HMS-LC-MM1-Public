I already described the modifications made in the extended round in the 1st round report. See "4. Potential Algorithm Improvements". These were:
 - read only the radiomics_gtv tag
 - using SegNet (bigger) model
 - generating data artificially (flipping, rotating, translating and adding Gaussian noise)

They are simple things, but each one of them take several hours to train and validate.
The most important thing I found from the 1st/extended round, is that, it is more important to focus in pre-processing the data than to modify the deep learning model.