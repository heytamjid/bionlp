import os
from google import genai


def list_active_caches():
    # Initialize the client exactly as you do in your main annotation script
    client = genai.Client(
        vertexai=True, project="project-ade5f3ce-e086-4d4c-91c", location="global"
    )

    print("Fetching active caches...\n")

    try:
        # Fetch the list of caches
        caches = list(client.caches.list())

        if not caches:
            print("No active caches found.")
            return

        # Print cleanly formatted details for each cache
        for cache in caches:
            print(f"Cache ID:     {cache.name}")
            print(f"Display Name: {cache.display_name}")
            print(f"Model:        {cache.model}")
            print(f"Expires:      {cache.expire_time}")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred while fetching caches:\n{e}")

    client.caches.delete(
        "projects/942972453935/locations/global/cachedContents/7649212818599706624"
    )


if __name__ == "__main__":
    list_active_caches()
