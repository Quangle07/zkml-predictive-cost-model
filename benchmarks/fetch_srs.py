import asyncio
import ezkl
import os

async def main():
    # Ensure the EZKL SRS cache directory exists
    os.makedirs("/home/hle2/.ezkl/srs", exist_ok=True)

    print("Downloading KZG 18...")
    await ezkl.get_srs(logrows=18)

    print("Downloading KZG 19...")
    await ezkl.get_srs(logrows=19)

    print("SRS Download Complete.")

if __name__ == "__main__":
    asyncio.run(main())
