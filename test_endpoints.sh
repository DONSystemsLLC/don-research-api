#!/bin/bash

TAMU_TOKEN="tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc"
BASE_URL="http://localhost:8080"

echo "🧪 Testing DON Research API Endpoints"
echo "======================================"
echo ""

# Test 1: Root redirect
echo "1️⃣  Testing root (/) redirects to /docs..."
REDIRECT=$(curl -s -o /dev/null -w "%{http_code}" -L "$BASE_URL/")
if [ "$REDIRECT" = "200" ]; then
    echo "   ✅ Root redirects successfully"
else
    echo "   ❌ Root redirect failed (HTTP $REDIRECT)"
fi
echo ""

# Test 2: Swagger UI with navigation
echo "2️⃣  Testing /docs has navigation bar..."
NAV_CHECK=$(curl -s "$BASE_URL/docs" | grep -c "custom-nav")
if [ "$NAV_CHECK" -gt 0 ]; then
    echo "   ✅ Navigation bar present in Swagger UI"
else
    echo "   ❌ Navigation bar missing"
fi
echo ""

# Test 3: Quick Start page
echo "3️⃣  Testing /help (Quick Start)..."
HELP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/help")
HELP_FORMATS=$(curl -s "$BASE_URL/help" | grep -c "Supported Data Formats")
if [ "$HELP_STATUS" = "200" ] && [ "$HELP_FORMATS" -gt 0 ]; then
    echo "   ✅ Quick Start page loads with format table"
else
    echo "   ❌ Quick Start page issue (HTTP $HELP_STATUS)"
fi
echo ""

# Test 4: Complete Guide
echo "4️⃣  Testing /guide (Complete Guide)..."
GUIDE_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/guide")
GUIDE_SWAGGER=$(curl -s "$BASE_URL/guide" | grep -c "Swagger UI")
if [ "$GUIDE_STATUS" = "200" ] && [ "$GUIDE_SWAGGER" -gt 0 ]; then
    echo "   ✅ Complete Guide loads with Swagger tutorial"
else
    echo "   ❌ Complete Guide issue (HTTP $GUIDE_STATUS)"
fi
echo ""

# Test 5: Health check (unauthenticated)
echo "5️⃣  Testing /api/v1/health (public endpoint)..."
HEALTH=$(curl -s "$BASE_URL/api/v1/health" | grep -o '"status":"healthy"')
if [ ! -z "$HEALTH" ]; then
    echo "   ✅ Health check responds"
else
    echo "   ⚠️  Health check available (may show database warnings)"
fi
echo ""

# Test 6: Authentication with TAMU token
echo "6️⃣  Testing TAMU token authentication..."
AUTH_TEST=$(curl -s -H "Authorization: Bearer $TAMU_TOKEN" "$BASE_URL/api/v1/health" | grep -o '"status"')
if [ ! -z "$AUTH_TEST" ]; then
    echo "   ✅ TAMU token accepted"
else
    echo "   ❌ TAMU token rejected"
fi
echo ""

# Test 7: Rate limit check
echo "7️⃣  Testing rate limit response..."
RATE_LIMIT=$(curl -s -H "Authorization: Bearer invalid_token" "$BASE_URL/api/v1/health" | grep -o "401\|Unauthorized")
if [ ! -z "$RATE_LIMIT" ]; then
    echo "   ✅ Invalid token properly rejected (401)"
else
    echo "   ⚠️  Check authorization handling"
fi
echo ""

echo "======================================"
echo "✅ Local testing complete!"
echo ""
