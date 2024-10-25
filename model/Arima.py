import numpy as np
import torch
from pmdarima.arima import auto_arima
from tqdm import tqdm

class ARIMA():
    def __init__(self,seq_len=96,pred_len = 96):
        super(ARIMA, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len

    def forward(self, enc_x):
        #--------------------------------------
        #   enc_x[batch,seq,dim]
        #--------------------------------------

        S,D = enc_x.shape
        x = enc_x
        for j in tqdm(range(D)):
            ts = x[-365:,j]
            pre = []

            for m in range(1):
                ts = np.expand_dims(ts,axis = 1)

                '''
                Explanation of some parameters of auto_arima.
                    1. start_p: the starting value of p, the order (or number of lags) of the autoregressive (‘AR’) model, must be a positive integer.
                    2. start_q: the initial value of q, the order of the moving average (‘MA’) model. Must be a positive integer.
                    3. max_p: the maximum value of p, must be a positive integer greater than or equal to start_p.
                    4. max_q:the maximum value of q, must be a positive integer greater than start_q
                    5. seasonal:whether seasonal ARIMA is appropriate. default is true. Note that if season is true and m == 1, season will be set to False.
                    6.stationary :Whether the time series is smooth and d is zero.
                    6.information_criterion : Information criterion is used to select the best ARIMA model. One of (‘aic’, ‘bic’, ‘hqic’, ‘oob’)
                    7. alpha: test level of test significance, default 0.05
                    8. test: if stationary is false and d is None, the type of unit root test used to detect smoothness. Default is ‘kpss’; can be set to adf
                    9. n_jobs : number of models fitted in parallel in the lattice search (progressive = False). Default is 1, but -1 can be used to indicate ‘as many as possible’.
                    10. suppress_warnings: many warnings may be thrown in the statsmodel. If suppress_warnings is true, then all warnings from ARIMA will be suppressed.
                    11. error_action: if for some reason ARIMA cannot be matched, the error handling behaviour can be controlled. (warn,raise,ignore,trace)
                    12. max_d:Maximum value of d, i.e. maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
                    13. trace:Whether to print the fitness status. A value of False will not print any debugging information. A value of True will print some
                '''
                model = auto_arima(ts, start_p=0, start_q=0, max_p=5, max_q=12, max_d=12,
                                   seasonal=True, test='adf',
                                   error_action='ignore',
                                   information_criterion='aic',
                                   njob=-1, suppress_warnings=True)
                model.fit(ts)
                forecast = model.predict(n_periods=self.pred_len)
                pre.append(forecast)
            predictions = torch.tensor(np.array(pre).reshape(-1, 1))



            if j == 0:
                pred = predictions
                continue
            pred = torch.cat([pred,predictions], dim=-1)

        return pred