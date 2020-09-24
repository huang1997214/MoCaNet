import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="MAGCAPNN")
    parser.add_argument("data_dir",
                        nargs="?",
                        default="./data",
                        help="Data dir.")

    parser.add_argument("log_dir",
                        nargs="?",
                        default="./log",
                        help="Log dir.")

    parser.add_argument("data_name",
                        nargs="?",
                        default="ENZYMES",
                        help="Data name.")

    parser.add_argument("--gcn_filters",
                        type=list,
                        default=[15, 15, 15],
                        help="Number of Graph Convolutional filters. Default is [20, 20].")

    parser.add_argument("--motif_num",
                        type=int,
                        default=2,
                        help="Number of Different Motifs.")

    parser.add_argument("--capsule_dimensions",
                        type=int,
                        default=15,
                        help="Capsule dimensions. Default is 8.")

    parser.add_argument("--Pri_Cap_Num",
                        type=int,
                        default=3,
                        help="Number of primary capsule.")

    parser.add_argument("--Super_Node_Dim",
                        type=int,
                        default=15,
                        help="Super Node Dimension")

    parser.add_argument("--Super_Node_Num",
                        type=int,
                        default=25,
                        help="Super Node NUm")

    parser.add_argument("--Class_Capsule_Num",
                        type=int,
                        default=6,
                        help="Capsule num in class capsule layer(Equal to the class num)")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.0005.")

    parser.add_argument("--lambd",
                        type=float,
                        default=0.5,
                        help="Loss combination weight. Default is 0.5.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10 ** -6,
                        help="Weight decay. Default is 10^-6.")

    parser.add_argument("--theta",
                        type=float,
                        default=0.0005,
                        help="Reconstruction loss weight. Default is 0.001.")

    parser.add_argument("--Use_Motif",
                        default=True,
                        help="Use motif or not")

    parser.add_argument("--Motif_threshold",
                        type=int,
                        default=4,
                        help="The threshold of the edges in motif adj matrix")

    parser.add_argument("--Epoch",
                        type=int,
                        default=400,
                        help="Epoch")

    parser.add_argument("--Use_Recon",
                        default=False,
                        help="Use Reconstruction loss or not")

    parser.add_argument("--Recon_type",
                        default='MY',
                        help="Reconstruction Loss Type")

    parser.add_argument("--Core_node_num",
                        type=int,
                        default=18,
                        help="Reconstruction Node")

    parser.add_argument("--Recon_vis",
                        default=False,
                        help="Show Reconstruction Result Or Not")
    return parser.parse_args()