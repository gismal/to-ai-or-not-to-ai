import pytest

@pytest.mark.anyio
async def test_create_feedback_in_db(feedback_repo, mock_feedback_data):
    # Save the data
    await feedback_repo.create_feedback(
        filename=mock_feedback_data.filename,
        model_prediction=mock_feedback_data.model_prediction,
        confidence=mock_feedback_data.confidence,
        user_correction=mock_feedback_data.user_correction.value if hasattr(mock_feedback_data.user_correction, 'value') else mock_feedback_data.user_correction
    )
    
    # read the data and confirm
    db_item = await feedback_repo.get_feedback_by_filename("db_test_image.png")
    
    assert db_item is not None
    assert db_item.confidence == 0.99
    assert db_item.model_prediction == "AI_GENERATED"