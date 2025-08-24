"use client";
import { Input } from "@heroui/input";
import { Alert } from "@heroui/alert";
import { title, subtitle } from "@/components/primitives";
import { useState } from "react";
import { RequestType } from "@/components/request_type";
import { extractid } from "@/components/id_extraction";
import { Button, ButtonGroup } from "@heroui/button";
import axios from "axios";

export default function Home() {
  const [request, setRequest] = useState<RequestType>({
    source_folder_id: null,
    result_folder_id: null,
    ai_model_name: null,
    api_key: null,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const API_URL: string =
    "https://handwritten-document-processing.onrender.com";

  //Test first
  const onSubmit = async () => {
    setError(null);
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/run-pipeline`, request);
      console.log("Response Data:", response.data);
    } catch (err: any) {
      let message = "An unexpected error occurred.";
      if (err.response && err.response.data && err.response.data.error) {
        message = err.response.data.error;
      } else if (err.message) {
        message = err.message;
      }
      setError(message);
      console.error("Error fetching data:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
      <div className="inline-block max-w-5xl text-center justify-center">
        <span
          className={
            title() +
            " bg-gradient-to-r from-pink-500 to-blue-500 text-transparent bg-clip-text"
          }
        >
          Historical Document Processing
        </span>
        <div className="my-3 flex flex-col items-center">
          <span className={subtitle() + " text-cyan-700"}>
            Unlock insights from your documents
          </span>
          <span className={subtitle() + " text-cyan-800"}>
            with our powerful AI tools
          </span>
          <span className={subtitle() + " text-cyan-900 font-semibold"}>
            For the AUC library
          </span>
        </div>
        {error && (
          <div className="w-8/12 mx-auto mb-2">
            <Alert
              color="danger"
              variant="bordered"
              title="Error"
              onClose={() => setError(null)}
            >
              {error}
            </Alert>
          </div>
        )}
        {loading && (
          <div className="w-8/12 mx-auto mb-2">
            <Alert color="primary" variant="bordered" title="Processing...">
              <div className="flex items-center gap-2">
                <svg
                  className="animate-spin h-5 w-5 text-blue-500"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                  ></path>
                </svg>
                <span>
                  This may take several minutes (or up to an hour) to complete.
                  Please do not close this page.
                </span>
              </div>
            </Alert>
          </div>
        )}
        <div className="flex flex-col items-center gap-4 mt-9">
          <div className="w-8/12 mx-auto hover:w-9/12 transition-all duration-300">
            <Input
              name="api_key"
              type="text"
              value={request.api_key || ""}
              onChange={(e) => {
                setRequest({ ...request, api_key: e.target.value });
                console.log("API Key:", e.target.value);
              }}
              placeholder="Paste your API Key"
              variant="bordered"
              label="API Key"
              radius="md"
            />
          </div>
          <div className="w-8/12 mx-auto hover:w-9/12 transition-all duration-300">
            <Input
              name="ai_model_name"
              type="text"
              value={request.ai_model_name || ""}
              onChange={(e) => {
                setRequest({ ...request, ai_model_name: e.target.value });
                console.log("AI Model Name:", e.target.value);
              }}
              placeholder="Example: gpt-5-2025-08-07"
              variant="bordered"
              label="OpenAi Model"
              radius="md"
            />
          </div>
          <div className="w-8/12 mx-auto hover:w-9/12 transition-all duration-300">
            <Input
              name="source_folder_url"
              type="text"
              value={request.source_folder_id || ""}
              onChange={(e) => {
                setRequest({
                  ...request,
                  source_folder_id: extractid(e.target.value),
                });
                console.log("Source Folder ID:", extractid(e.target.value));
              }}
              placeholder="Example: https://drive.google.com/drive/folders/1loLg-htSD0XtU5MgzofbVCd4lFMsiKpg?dmr=1&ec=wgc-drive-hero-goto"
              variant="bordered"
              label="Source Folder Url"
              radius="md"
            />
          </div>
          <div className="w-8/12 mx-auto hover:w-9/12 transition-all duration-300">
            <Input
              name="result_folder_url"
              type="text"
              value={request.result_folder_id || ""}
              onChange={(e) => {
                setRequest({
                  ...request,
                  result_folder_id: extractid(e.target.value),
                });
                console.log("Result Folder ID:", extractid(e.target.value));
              }}
              placeholder="Example: https://drive.google.com/drive/folders/1NN8_VERmh4xe0Z2mZiQSqYkwxmj5fNRs?dmr=1&ec=wgc-drive-hero-goto"
              variant="bordered"
              label="Result Folder Url"
              radius="md"
            />
          </div>
          <div className="w-8/12 mx-auto hover:w-9/12 transition-all duration-300">
            <ButtonGroup>
              <Button onPress={onSubmit} isLoading={loading}>
                Submit
              </Button>
              <Button
                variant="faded"
                onPress={() =>
                  setRequest({
                    source_folder_id: null,
                    result_folder_id: null,
                    ai_model_name: null,
                    api_key: null,
                  })
                }
              >
                Reset
              </Button>
            </ButtonGroup>
          </div>
        </div>
      </div>
    </section>
  );
}
