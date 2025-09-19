import jwt
import os
from typing import Optional, Dict
from datetime import datetime

class TokenValidator:
    def __init__(self):
        # ===== DEV MODE CONFIGURATION =====
        # Set to True to bypass token validation in development
        # IMPORTANT: Must be False in production!
        self.DEV_MODE = os.getenv('AUTH_DEV_MODE', 'false').lower() == 'true'
        
        if self.DEV_MODE:
            print("⚠️  WARNING: Auth DEV MODE is enabled - tokens will not be validated!")
            print("⚠️  This should ONLY be used in development!")
        # ===== END DEV MODE =====
        
        self.secret = os.getenv('BACKEND_SHARED_SECRET')
        if not self.secret and not self.DEV_MODE:
            raise ValueError("BACKEND_SHARED_SECRET not configured (required when not in dev mode)")
    
    def validate_token(self, token: str) -> Optional[Dict]:
        """Validate JWT token from backend"""
        
        # ===== DEV MODE BYPASS =====
        if self.DEV_MODE:
            # Return mock session data for development
            print(f"[DEV MODE] Bypassing token validation")
            return {
                'attempt_id': 'dev-attempt-123',
                'student_id': 'dev-student-456',
                'correlation_token': f'dev_correlation_{datetime.now().timestamp()}',
                'expires_at': int(datetime.now().timestamp()) + 3600
            }
        # ===== END DEV MODE BYPASS =====
        
        # Production token validation
        try:
            payload = jwt.decode(
                token, 
                self.secret,
                algorithms=['HS256']
            )
            
            # Additional validation
            if payload.get('type') != 'voice_session':
                raise jwt.InvalidTokenError("Invalid token type")
            
            return {
                'attempt_id': payload.get('attemptId'),
                'student_id': payload.get('studentId'),
                'correlation_token': payload.get('correlationToken'),
                'expires_at': payload.get('exp')
            }
            
        except jwt.ExpiredSignatureError:
            print("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            print(f"Invalid token: {e}")
            return None

token_validator = TokenValidator()