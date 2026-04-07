Download data from:

https://opus.nlpl.eu/OpenSubtitles/en&es/v2024/OpenSubtitles


Frontend:
source ~/.nvm/nvm.sh && nvm use 20 && npm run dev -- --port 4000

Backend:
.venv/bin/uvicorn backend:app --port 9000 --reload



Will support 4 main ways of doing arithmetics the following:

1. Function like:

    std function        => sum(3,4)
    with no parenthesis => times 8 8

2. operand like:

    natural language    => 3 plus 4
    with spaces         => 3     times    3

3. Object like:

    calling method           => 3.sum(4)
    calling method, no paren => 3.times 4 

4. calculator style:

    std arithmetics in many langs => 3 + 4
    std arithmetics in many langs => 3 * 4
