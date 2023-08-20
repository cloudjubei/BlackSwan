from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)

# TODO