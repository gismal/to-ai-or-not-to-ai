import pytest

@pytest.mark.asyncio
async def test_create_feedback_in_db(feedback_repo, mock_feedback_data):
    # Save the data
    await feedback_repo.create_feedback(
        filename=mock_feedback_data.filename,
        model_prediction=mock_feedback_data.model_prediction,
        confidence=mock_feedback_data.confidence,
        user_correction=mock_feedback_data.user_correction
    )
    
    # read the data and confirm
    db_item = await feedback_repo.get_feedback_by_filename("db_test_image.png")
    
    assert db_item is not None
    assert db_item.confidence == 0.99
    assert db_item.model_prediction == "AI_GENERATED"
    assert db_item.filename       == "db_test_image.png"
    assert db_item.user_correction  == FeedbackLabel.REAL
    assert db_item.error_type       == ErrorType.FALSE_POSITIVE  # AI_GENERATED + REAL correction
    assert db_item.is_deleted       == False
    
@pytest.mark.asyncio
async def test_soft_delete(feedback_repo, mock_feedback_data):
    await feedback_repo.create_feedback(...)
    item = await feedback_repo.get_feedback_by_filename("db_test_image.png")
        
    deleted = await feedback_repo.soft_delete(item.id)
    
    assert deleted is True
    # now, soft deleted items should be listed in get_recent_errors
    errors = await feedback_repo.get_recent_errors()
    assert all(e.id != item.id for e in errors)
    
@pytest.mark.asyncio 
async def test_get_recent_errors_pagination(feedback_repo, mock_feedback_data):
    for _ in range(3):
        await feedback_repo.create_feedback(...)
        
   page_1 = await feedback_repo.get_recent_errors(limit=2, offset=0)
   page_2 = await feedback_repo.get_recent_errors(limit=2, offset=2)
    
   assert len(page_1) == 2
   assert len(page_2) == 1

@pytest.mark.asyncio
async def test_get_feedback_by_filename_not_found(feedback_repo):
   result = await feedback_repo.get_feedback_by_filename("nonexistent.png")
   assert result is None