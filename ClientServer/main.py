#pip
from fastapi import Depends, FastAPI, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import uvicorn
import uuid

# Eternal files
from config import *
from entities import *
from db import *


#
app = FastAPI(root_path="/")

# TODO: develop a security policy afterwards for browser rendering
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    #allow_origins=["ручка до фронта"]
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Фейковая база пользователей
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
    }
}

# Хранилище refresh-токенов (в реальном проекте — Redis/БД)
refresh_tokens_store = {}


#logic
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str) -> dict:
    return fake_users_db.get(username)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + expires_delta})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + expires_delta})
    return jwt.encode(to_encode, REFRESH_SECRET_KEY, algorithm=ALGORITHM)


#UTILS
def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        return "expired"
    except jwt.PyJWTError:
        return None

def decode_refresh_token(token: str):
    try:
        payload = jwt.decode(token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:
        return None


#Routes
@app.get("/")
async def root():
    return {"success": True}


'''
@app.get("/cabinet")
async def cabinet(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user(username)
    if user is None or user.get("disabled", False):
        raise HTTPException(status_code=403, detail="User is disabled")

    return user
'''


@app.get("/cabinet")
async def cabinet(request: Request, response: Response, token: str = Depends(oauth2_scheme)):
    username = decode_access_token(token)

    if username == "expired":
        refresh_token = request.headers.get("X-Refresh-Token")
        if not refresh_token:
            raise HTTPException(status_code=401, detail="Access token expired. Provide refresh token.")
        refresh_username = decode_refresh_token(refresh_token)
        if not refresh_username or refresh_token not in refresh_tokens_store:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Issue new tokens
        new_access = create_access_token({"sub": refresh_username})
        new_refresh = create_refresh_token({"sub": refresh_username, "jti": str(uuid.uuid4())})
        del refresh_tokens_store[refresh_token]
        refresh_tokens_store[new_refresh] = refresh_username

        response.headers["X-New-Access-Token"] = new_access
        response.headers["X-New-Refresh-Token"] = new_refresh
        username = refresh_username

    elif username is None:
        raise HTTPException(status_code=401, detail="Invalid access token")

    user = get_user(username)
    if not user or user.get("disabled", False):
        raise HTTPException(status_code=403, detail="User is disabled")

    return user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"], "jti": str(uuid.uuid4())})

    # Сохраняем refresh token в памяти (в реальном проекте это Redis/БД)
    refresh_tokens_store[refresh_token] = user["username"]

    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


"""
@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user(username)
    if user is None or user.get("disabled", False):
        raise HTTPException(status_code=403, detail="User is disabled")

    return user
"""

@app.get("/users/me")
async def read_users_me(request: Request, response: Response, token: str = Depends(oauth2_scheme)):
    username = decode_access_token(token)

    if username == "expired":
        refresh_token = request.headers.get("X-Refresh-Token")
        if not refresh_token:
            raise HTTPException(status_code=401, detail="Access token expired. Provide refresh token.")
        refresh_username = decode_refresh_token(refresh_token)
        if not refresh_username or refresh_token not in refresh_tokens_store:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        # Issue new tokens
        new_access = create_access_token({"sub": refresh_username})
        new_refresh = create_refresh_token({"sub": refresh_username, "jti": str(uuid.uuid4())})
        del refresh_tokens_store[refresh_token]
        refresh_tokens_store[new_refresh] = refresh_username

        response.headers["X-New-Access-Token"] = new_access
        response.headers["X-New-Refresh-Token"] = new_refresh
        username = refresh_username

    elif username is None:
        raise HTTPException(status_code=401, detail="Invalid access token")

    user = get_user(username)
    if not user or user.get("disabled", False):
        raise HTTPException(status_code=403, detail="User is disabled")

    return user


@app.post("/logout")
async def logout(refresh_data: RefreshTokenRequest):
    # Проверяем, есть ли refresh токен в хранилище
    if refresh_data.refresh_token not in refresh_tokens_store:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Удаляем refresh токен
    del refresh_tokens_store[refresh_data.refresh_token]

    return {"message": "Successfully logged out"}


@app.get("/healthcheck")
async def healthcheck():
    return {"success": True}


#Entry Point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
