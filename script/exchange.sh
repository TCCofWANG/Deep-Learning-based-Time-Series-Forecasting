
for pred_Len in 48 96 336
do
for model_name in autoformer Fedformer LSTnet Deepar TCN informer TDformer \
reformer logtrans CNN_1D GRU_RNN SAE Autoencoder Deepssm Pyraformer \
Aliformer Transformer Nbeat deep_states ETSformer PatchTST Scaleformer DLinear \
Crossformer Triformer NS_Transformer koopa iTransformer FITS TimeMixer
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='deep_learning'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done

for pred_Len in 48 96 336
do
for model_name in SSD
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='deep_learning'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=5 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done

for pred_Len in 48 96 336
do
for model_name in TFT
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='tft'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done

for pred_Len in 48 96 336
do
for model_name in Arima
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='arima'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done

for pred_Len in 48 96 336
do
for model_name in WSAES_LSTM
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='wases'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done

for pred_Len in 48 96 336
do
for model_name in AST
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='gan'\
  --seed=1 \
  --data_name=exchange \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done