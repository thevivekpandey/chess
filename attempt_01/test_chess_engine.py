"""
Tests for the chess engine neural network.
"""

import torch
import numpy as np
from chess_engine import (
    fen_to_tensor,
    normalize_eval,
    denormalize_eval,
    ChessNet,
    ResidualBlock,
    ChessDataset,
    create_sample_data,
    NUM_PLANES,
    PIECE_TO_PLANE
)


def test_fen_to_tensor_starting_position():
    """Test encoding of the starting position."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen)

    assert tensor.shape == (18, 8, 8), f"Expected shape (18, 8, 8), got {tensor.shape}"

    # Check white pawns on rank 2 (row 6 in 0-indexed from top)
    white_pawn_plane = tensor[PIECE_TO_PLANE['P']]
    assert white_pawn_plane[6, :].sum() == 8, "Should have 8 white pawns on rank 2"

    # Check black pawns on rank 7 (row 1 in 0-indexed from top)
    black_pawn_plane = tensor[PIECE_TO_PLANE['p']]
    assert black_pawn_plane[1, :].sum() == 8, "Should have 8 black pawns on rank 7"

    # Check white king on e1
    white_king_plane = tensor[PIECE_TO_PLANE['K']]
    assert white_king_plane[7, 4] == 1.0, "White king should be on e1"

    # Check black king on e8
    black_king_plane = tensor[PIECE_TO_PLANE['k']]
    assert black_king_plane[0, 4] == 1.0, "Black king should be on e8"

    # Check castling rights (all should be 1 for starting position)
    assert tensor[12].sum() == 64, "White kingside castling should be all 1s"
    assert tensor[13].sum() == 64, "White queenside castling should be all 1s"
    assert tensor[14].sum() == 64, "Black kingside castling should be all 1s"
    assert tensor[15].sum() == 64, "Black queenside castling should be all 1s"

    # Check side to move (white)
    assert tensor[16].sum() == 64, "Side to move should be white (all 1s)"

    # Check no en passant
    assert tensor[17].sum() == 0, "No en passant square in starting position"

    print("test_fen_to_tensor_starting_position PASSED")


def test_fen_to_tensor_with_en_passant():
    """Test encoding with en passant square."""
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    tensor = fen_to_tensor(fen)

    # Check en passant on e3 (row 5, col 4)
    assert tensor[17, 5, 4] == 1.0, "En passant should be on e3"
    assert tensor[17].sum() == 1, "Only one en passant square should be marked"

    # Check side to move (black)
    assert tensor[16].sum() == 0, "Side to move should be black (all 0s)"

    print("test_fen_to_tensor_with_en_passant PASSED")


def test_fen_to_tensor_no_castling():
    """Test encoding with no castling rights."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"
    tensor = fen_to_tensor(fen)

    # Check no castling rights
    assert tensor[12].sum() == 0, "No white kingside castling"
    assert tensor[13].sum() == 0, "No white queenside castling"
    assert tensor[14].sum() == 0, "No black kingside castling"
    assert tensor[15].sum() == 0, "No black queenside castling"

    print("test_fen_to_tensor_no_castling PASSED")


def test_fen_to_tensor_partial_castling():
    """Test encoding with partial castling rights."""
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w Kq - 0 1"
    tensor = fen_to_tensor(fen)

    # Check partial castling rights
    assert tensor[12].sum() == 64, "White kingside castling"
    assert tensor[13].sum() == 0, "No white queenside castling"
    assert tensor[14].sum() == 0, "No black kingside castling"
    assert tensor[15].sum() == 64, "Black queenside castling"

    print("test_fen_to_tensor_partial_castling PASSED")


def test_normalize_denormalize_eval():
    """Test evaluation normalization round-trip."""
    # Test values in pawn units (not centipawns)
    test_values = [-50, -10, -5, -1, 0, 1, 5, 10, 50]

    for pawns in test_values:
        normalized = normalize_eval(pawns)
        assert -1.0 <= normalized <= 1.0, f"Normalized value {normalized} out of range"

        denormalized = denormalize_eval(normalized)
        assert abs(denormalized - pawns) < 0.01, f"Round-trip error: {pawns} -> {normalized} -> {denormalized}"

    print("test_normalize_denormalize_eval PASSED")


def test_residual_block():
    """Test residual block dimensions."""
    block = ResidualBlock(channels=512)
    x = torch.randn(4, 512, 8, 8)
    out = block(x)

    assert out.shape == x.shape, f"Residual block should preserve shape, got {out.shape}"
    print("test_residual_block PASSED")


def test_chess_net_forward():
    """Test ChessNet forward pass."""
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)
    x = torch.randn(4, NUM_PLANES, 8, 8)
    out = model(x)

    assert out.shape == (4, 1), f"Expected output shape (4, 1), got {out.shape}"
    assert (out >= -1).all() and (out <= 1).all(), "Output should be in [-1, 1] range"

    print("test_chess_net_forward PASSED")


def test_chess_net_parameter_count():
    """Test that the model has a reasonable number of parameters."""
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)
    num_params = sum(p.numel() for p in model.parameters())

    # Rough estimate: 256 channels, 8 res blocks with 2 conv each
    # Should be around 10M parameters
    assert num_params > 1_000_000, "Model should have at least 1M parameters"
    assert num_params < 50_000_000, "Model should have less than 50M parameters"

    print(f"test_chess_net_parameter_count PASSED (params: {num_params:,})")


def test_chess_dataset():
    """Test ChessDataset."""
    data = create_sample_data()
    dataset = ChessDataset(data)

    assert len(dataset) == len(data), "Dataset length should match data length"

    board, eval_tensor = dataset[0]
    assert board.shape == (NUM_PLANES, 8, 8), f"Board shape should be ({NUM_PLANES}, 8, 8)"
    assert eval_tensor.shape == (1,), "Eval tensor shape should be (1,)"
    assert -1 <= eval_tensor.item() <= 1, "Eval should be normalized to [-1, 1]"

    print("test_chess_dataset PASSED")


def test_full_pipeline():
    """Test the full pipeline: encoding -> model -> output."""
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)
    model.eval()

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = fen_to_tensor(fen).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(tensor)

    assert output.shape == (1, 1), f"Output shape should be (1, 1), got {output.shape}"
    assert -1 <= output.item() <= 1, f"Output {output.item()} should be in [-1, 1]"

    print("test_full_pipeline PASSED")


def test_different_positions():
    """Test encoding of various positions."""
    positions = [
        # Empty board (just kings)
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        # Position after 1.e4 e5 2.Nf3 Nc6 3.Bb5
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        # Endgame
        "8/8/4k3/8/8/3K4/4P3/8 w - - 0 1",
        # Complex middlegame
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
    ]

    for fen in positions:
        tensor = fen_to_tensor(fen)
        assert tensor.shape == (NUM_PLANES, 8, 8), f"Shape mismatch for FEN: {fen}"

        # Check that piece planes sum to reasonable values
        piece_sum = tensor[:12].sum()
        assert 2 <= piece_sum <= 32, f"Unreasonable piece count for FEN: {fen}"

    print("test_different_positions PASSED")


def test_gradient_flow():
    """Test that gradients flow properly through the network."""
    model = ChessNet(initial_channels=512, res_channels=256, num_res_blocks=8)
    x = torch.randn(2, NUM_PLANES, 8, 8, requires_grad=True)
    target = torch.randn(2, 1)

    output = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    print("test_gradient_flow PASSED")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running chess engine tests...")
    print("=" * 60)

    test_fen_to_tensor_starting_position()
    test_fen_to_tensor_with_en_passant()
    test_fen_to_tensor_no_castling()
    test_fen_to_tensor_partial_castling()
    test_normalize_denormalize_eval()
    test_residual_block()
    test_chess_net_forward()
    test_chess_net_parameter_count()
    test_chess_dataset()
    test_full_pipeline()
    test_different_positions()
    test_gradient_flow()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
