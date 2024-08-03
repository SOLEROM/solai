# fscoreSupport


```
# Calculating the Scores
vHatY                    = oSVM.predict(mX)
precision, recall, f1, support = precision_recall_fscore_support(vY, vHatY, pos_label = 1, average = 'binary')


print(f'Precision = {precision:0.3f}')
print(f'Recall    = {recall:0.3f}'   )
print(f'f1        = {f1:0.3f}'       )
print(f'Support   = {support}'  )

```