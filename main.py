import argparse
from train import train_model
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Guided Contrastive Graph Transformer")
    parser.add_argument('--mode', type=str, choices=['train', 'infer', 'ui'], required=True, 
                        help='Mode to run')
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\jayak\Downloads\All_faces_sculpted_derivatives', help='Dataset directory')
    parser.add_argument('--mesh', type=str, default=None, help='Mesh file for inference')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args.data_dir)
    elif args.mode == 'infer':
        if args.mesh is None:
            print("Please provide a mesh file using --mesh for inference mode.")
            return
        stats, out_path = run_inference(args.mesh)
        print("Inference completed.")
        print(stats)
    elif args.mode == 'ui':
        import app
        app.app.launch(server_name="127.0.0.1")

if __name__ == "__main__":
    main()
