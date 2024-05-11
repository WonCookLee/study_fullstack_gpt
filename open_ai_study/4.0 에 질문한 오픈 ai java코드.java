import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;

public class OpenAIClient {
    private final String apiKey;

    public OpenAIClient(String apiKey) {
        this.apiKey = apiKey;
    }

    public String callGPT3(String prompt) throws IOException {
        OkHttpClient client = new OkHttpClient();

        MediaType mediaType = MediaType.parse("application/json");
        String jsonBody = "{\"prompt\": \"" + prompt + "\", \"max_tokens\": 150}";
        RequestBody body = RequestBody.create(jsonBody, mediaType);

        Request request = new Request.Builder()
                .url("https://api.openai.com/v1/engines/davinci/completions")
                .post(body)
                .addHeader("Content-Type", "application/json")
                .addHeader("Authorization", "Bearer " + this.apiKey)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) throw new IOException("Unexpected code " + response);

            return response.body().string();
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java OpenAIClient <api_key>");
            System.exit(1);
        }

        String apiKey = args[0];
        OpenAIClient client = new OpenAIClient(apiKey);

        try {
            String response = client.callGPT3("Hello, world!");
            System.out.println("Response from GPT-3: " + response);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}