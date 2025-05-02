#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mobilevit import MobileViT
from carla_dataset import CarlaDataset  # Assume we've created this
import time

def train_model(data_dir, epochs, batch_size, lr, output_dir):
    print(f"Training with data_dir={data_dir}, batch_size={batch_size}, epochs={epochs}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MobileViT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Load dataset
    print("Loading dataset...")
    train_dataset = CarlaDataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset loaded with {len(train_dataset)} samples.")

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        batch_count = 0
        start_time = time.time()

        model.train()

        for step, (rgb, depth, _, _) in enumerate(train_loader):
            rgb, depth = rgb.to(device), depth.to(device)
            pred = model(rgb)
            loss = criterion(pred, depth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if step % 10 == 0:
                print(f"Step {step:03d}: batch_loss = {loss.item():.6f}")

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.1f} sec | Avg Loss = {avg_loss:.6f}")

        # Save model
        model_path = os.path.join(output_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

def run_simulation(model_path):
    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Train first!")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileViT().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Rest of simulation code...
    print("Simulation started with model:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "simulate"], required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-model", default="./models/mobilevit_depth.pth")
    parser.add_argument("--model-path", help="Path to model for simulation mode")
    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    
    if args.mode == "train":
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=os.path.dirname(args.output_model)
        )
    elif args.mode == "simulate":
        if not args.model_path:
            # Default to latest trained model if none specified
            args.model_path = args.output_model
        run_simulation(args.model_path)