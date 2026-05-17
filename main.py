import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="ASL Sign Language Demo Orchestrator")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--predict", action="store_true")
    
    parser.add_argument("--method", type=str, choices=['bayes', 'knn', 'rf'])
    parser.add_argument("--k", type=int, default=3)
    
    args = parser.parse_args()
    
    if args.extract:
        subprocess.run([sys.executable, "data_extractor.py"])
        
    elif args.evaluate:
        subprocess.run([sys.executable, "evaluate.py"])
        
    elif args.train:
        if not args.method:
            print("Error: --method is required")
            sys.exit(1)
        cmd = [sys.executable, "train.py", "--method", args.method, "--k", str(args.k)]
        subprocess.run(cmd)
        
    elif args.predict:
        if not args.method:
            print("Error: --method is required")
            sys.exit(1)
        cmd = [sys.executable, "predict.py", "--method", args.method, "--k", str(args.k)]
        subprocess.run(cmd)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
