import pandas as pd

from parser import parse_args
from MultiOmicNet import multiomicNet

def moNetImpute(**kwargs):

    args = parse_args()

    for key, value in kwargs.items():
        setattr(args, key, value)
    
    X = pd.read_csv(args.inputData, index_col=0)
    Y = pd.read_csv(args.targetData, index_col=0)

    if args.cell_axis_input == "columns":
        X = X.T
        
    if args.cell_axis_target == "columns":
        Y = Y.T

    NN_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'ncores': args.cores,
        'sub_outputdim': args.output_neurons,
        'architecture': [
            {"type": "dense", "activation": "relu", "neurons": args.hidden_neurons},
            {"type": "dropout", "activation": "dropout", "rate": args.dropout_rate}]
    }

    net = multiomicNet(**NN_params)
    net.fit(X, Y)

    imputed = net.predict(X, Y, imputed_only=False, policy=args.policy)

    if args.output is not None:
        imputed.to_csv(args.output)
    else:
        return imputed

if __name__ == "__main__":
    moNetImpute()
