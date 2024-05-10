import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import org.json.JSONObject;

public class OpenAiApi {

    private static final String API_KEY = "YOUR_API_KEY";
    private static final OkHttpClient client = new OkHttpClient();

    public static void main(String[] args) {
        String prompt = "Once upon a time";
        String completion = getCompletion(prompt);
        System.out.println(completion);
    }

    public static String getCompletion(String prompt) {
        String url = "https://api.openai.com/v1/engines/davinci/completions";
        JSONObject json = new JSONObject()
                .put("prompt", prompt)
                .put("max_tokens", 100);

        RequestBody body = RequestBody.create(json.toString(), MediaType.parse("application/json"));
        Request request = new Request.Builder()
                .url(url)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .post(body)
                .build();

        try {
            Response response = client.newCall(request).execute();
            String responseBody = response.body().string();
            JSONObject jsonResponse = new JSONObject(responseBody);
            return jsonResponse.getString("choices");
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}