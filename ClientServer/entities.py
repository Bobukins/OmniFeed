from pydantic import BaseModel
from typing import List, Literal, Union, Optional

#Data Schemes
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class InstaContainer(BaseModel):
    estimate: int
    reviews: int
    subscribers: int

class TwoGisContainer(BaseModel):
    reviews: int
    remoteness: int
    estimate: int


class Companies(BaseModel):
    id: int
    name: str
    estimate: int
    reviews_count: int
    remoteness: int
    city: str
    views: Optional[int]
    social_networks: dict[str, Union[InstaContainer, TwoGisContainer]]


class Blogers(BaseModel):
    id: int
    name: str
    city: str
    reposts: str
    comments: Optional[str]
    likes: Optional[str]
    emotional_phone: str
    views: Optional[int]
    social_networks: dict[str, Union[InstaContainer]]
    

class RefreshTokenRequest(BaseModel):
    refresh_token: str